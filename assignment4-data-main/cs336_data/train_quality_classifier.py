import gzip
import random
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
import time

from utils import (
    extract_text, identify_language, mask_emails, 
    mask_phone_numbers, mask_ips, classify_nsfw, 
    classify_toxic_speech, gopher_quality_filter
)

class QualityClassifierTrainer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for data and models
        self.wiki_urls_path = self.base_dir / "enwiki-20240420-extracted_urls.txt.gz"
        self.sampled_urls_path = self.base_dir / "sampled_positive_urls.txt"
        self.warc_path = self.base_dir / "positive_urls.warc.gz"
        self.training_data_path = self.base_dir / "quality_training_data.txt"
        self.model_path = self.base_dir / "quality_classifier.bin"
        
    def subsample_wikipedia_urls(self, sample_size: int = 50000) -> None:
        """
        Subsample URLs from Wikipedia extracted URLs file using reservoir sampling.
        """
        print(f"Subsampling {sample_size} URLs from Wikipedia references...")
        
        sample = []
        with gzip.open(self.wiki_urls_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(tqdm(f, desc="Sampling URLs")):
                url = line.strip()
                if not url or url.startswith('#'):
                    continue
                    
                if i < sample_size:
                    sample.append(url)
                else:
                    # Reservoir sampling
                    j = random.randint(0, i)
                    if j < sample_size:
                        sample[j] = url
        
        # Write sampled URLs
        with open(self.sampled_urls_path, 'w') as f:
            for url in sample:
                f.write(url + '\n')
        
        print(f"Sampled {len(sample)} URLs to {self.sampled_urls_path}")
    
    # def scrape_urls(self, timeout: int = 5) -> None:
    #     """
    #     Scrape URLs using wget to create WARC file.
    #     """
    #     print("Scraping URLs with wget...")
        
    #     cmd = [
    #         'wget',
    #         f'--timeout={timeout}',
    #         '-i', str(self.sampled_urls_path),
    #         f'--warc-file={self.warc_path.stem}',
    #         '-O', '/dev/null',
    #         '--warc-max-size=1G',
    #         '--tries=2',
    #         '--waitretry=1'
    #     ]
        
    #     # Run wget in the base directory
    #     subprocess.run(cmd, cwd=self.base_dir, check=True)
    #     print(f"WARC file created at {self.warc_path}")
    def scrape_urls_in_batches(self, batch_size: int = 5000, timeout: int = 1) -> None:
        """
        Scrape URLs in batches to avoid file descriptor limits.
        Shows real-time progress during scraping.
        """
        print("Scraping URLs in batches...")
        
        # Read all URLs
        with open(self.sampled_urls_path, 'r') as f:
            all_urls = [line.strip() for line in f if line.strip()]
        
        total_urls = len(all_urls)
        print(f"Total URLs to scrape: {total_urls}")
        print(f"Timeout per URL: {timeout} seconds")
        print(f"Batch size: {batch_size} URLs")
        
        # Process in batches
        num_batches = (total_urls + batch_size - 1) // batch_size
        
        overall_start_time = time.time()
        successful_batches = 0
        
        for i in range(0, total_urls, batch_size):
            batch_num = i // batch_size + 1
            batch_urls = all_urls[i:i + batch_size]
            
            print(f"\n{'='*60}")
            print(f"Processing batch {batch_num}/{num_batches}")
            print(f"URLs in this batch: {len(batch_urls)}")
            print(f"Progress: {i}/{total_urls} URLs processed so far")
            print(f"{'='*60}")
            
            # Create temporary file for this batch
            batch_file = self.base_dir / f"batch_{batch_num}_urls.txt"
            with open(batch_file, 'w') as f:
                for url in batch_urls:
                    f.write(url + '\n')
            
            # WARC file for this batch
            batch_warc_stem = f"positive_urls_batch_{batch_num}.warc"
            
            cmd = [
                'wget',
                f'--timeout={timeout}',
                '-i', str(batch_file),
                f'--warc-file={batch_warc_stem}',
                '-O', '/dev/null',
                '--warc-max-size=1G',
                '--tries=2',
                '--waitretry=1',
                '--progress=bar:force',  # Show progress bar
                '--show-progress'  # Display download progress
            ]
            
            batch_start_time = time.time()
            
            try:
                print(f"Running wget for batch {batch_num}...")
                print(f"Started at: {time.strftime('%H:%M:%S')}")
                
                # Run wget without capturing output so we can see real-time progress
                result = subprocess.run(
                    cmd, 
                    cwd=self.base_dir, 
                    check=False  # Don't raise exception on non-zero exit
                )
                
                batch_duration = time.time() - batch_start_time
                
                # Check if WARC file was created
                warc_file = self.base_dir / f"{batch_warc_stem}.gz"
                warc_exists = warc_file.exists()
                
                if warc_exists:
                    warc_size_mb = warc_file.stat().st_size / (1024 * 1024)
                    print(f"\n✓ Batch {batch_num} completed")
                    print(f"  - Duration: {batch_duration:.1f} seconds")
                    print(f"  - WARC file size: {warc_size_mb:.2f} MB")
                    print(f"  - Exit code: {result.returncode}")
                    successful_batches += 1
                else:
                    print(f"\n⚠ Batch {batch_num} completed but no WARC file created")
                    print(f"  - Duration: {batch_duration:.1f} seconds")
                    print(f"  - Exit code: {result.returncode}")
                
                # Estimate time remaining
                if batch_num < num_batches:
                    avg_time_per_batch = (time.time() - overall_start_time) / batch_num
                    remaining_batches = num_batches - batch_num
                    estimated_remaining = avg_time_per_batch * remaining_batches
                    print(f"  - Estimated time remaining: {estimated_remaining/60:.1f} minutes")
                    
            except Exception as e:
                print(f"\n⚠ Warning: Batch {batch_num} encountered an error: {e}")
                print("Continuing with next batch...")
            finally:
                # Clean up temporary batch file
                if batch_file.exists():
                    batch_file.unlink()
                    print(f"  - Cleaned up temporary file: {batch_file.name}")
        
        # Summary statistics
        total_duration = time.time() - overall_start_time
        
        # List all created WARC files
        warc_files = list(self.base_dir.glob("positive_urls_batch_*.warc.gz"))
        total_warc_size_mb = sum(f.stat().st_size for f in warc_files) / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"✓ SCRAPING COMPLETE!")
        print(f"{'='*60}")
        print(f"Total time: {total_duration/60:.1f} minutes")
        print(f"URLs processed: {total_urls}")
        print(f"Successful batches: {successful_batches}/{num_batches}")
        print(f"WARC files created: {len(warc_files)}")
        print(f"Total WARC size: {total_warc_size_mb:.2f} MB")
        print(f"Average speed: {total_urls/(total_duration/60):.1f} URLs/minute")
        print(f"{'='*60}\n")
    
    def filter_and_clean_text(self, text: str) -> Tuple[str, bool]:
        """
        Apply all quality filters to text.
        Returns: (cleaned_text, is_valid)
        """
        if not text:
            return "", False
        
        # Language identification
        lang, lang_conf = identify_language(text)
        if lang != 'en' or lang_conf < 0.7:
            return "", False
        
        # Mask PII
        text, _ = mask_emails(text)
        text, _ = mask_phone_numbers(text)
        text, _ = mask_ips(text)
        
        # NSFW filter
        nsfw_label, nsfw_conf = classify_nsfw(text)
        if nsfw_label == '__label__NSFW' and nsfw_conf > 0.5:
            return "", False
        
        # Toxic speech filter
        toxic_label, toxic_conf = classify_toxic_speech(text)
        if toxic_label == '__label__toxic' and toxic_conf > 0.5:
            return "", False
        
        # Gopher quality filter
        if not gopher_quality_filter(text):
            return "", False
        
        return text, True
    
    def extract_positive_examples(self, max_examples: int = 40000) -> List[str]:
        """
        Extract and filter positive examples from all WARC files.
        """
        print("Extracting positive examples from WARC files...")
        positive_examples = []
        
        # Find all WARC files (both single file and batched files)
        warc_files = list(self.base_dir.glob("positive_urls*.warc.gz"))
        
        if not warc_files:
            print("❌ No WARC files found!")
            print("Make sure scraping completed successfully.")
            return positive_examples
        
        print(f"Found {len(warc_files)} WARC file(s) to process")
        
        for warc_idx, warc_file in enumerate(warc_files, 1):
            print(f"\n{'='*60}")
            print(f"Processing WARC file {warc_idx}/{len(warc_files)}: {warc_file.name}")
            print(f"{'='*60}")
            
            try:
                with gzip.open(warc_file, 'rb') as stream:
                    records_processed = 0
                    records_valid = 0
                    
                    for record in tqdm(
                        ArchiveIterator(stream), 
                        desc=f"Processing {warc_file.name}",
                        unit="records"
                    ):
                        if record.record_type != WarcRecordType.response:
                            continue
                        
                        records_processed += 1
                        
                        try:
                            # Extract HTML content
                            html_bytes = record.reader.read()
                            text = extract_text(html_bytes)
                            
                            # Apply filters
                            cleaned_text, is_valid = self.filter_and_clean_text(text)
                            
                            if is_valid and cleaned_text:
                                positive_examples.append(cleaned_text)
                                records_valid += 1
                                
                                if len(positive_examples) >= max_examples:
                                    print(f"\n✓ Reached target of {max_examples} examples. Stopping.")
                                    print(f"Final stats for {warc_file.name}:")
                                    print(f"  - Records processed: {records_processed}")
                                    print(f"  - Valid examples: {records_valid}")
                                    print(f"  - Success rate: {records_valid/records_processed*100:.1f}%")
                                    return positive_examples
                        
                        except Exception as e:
                            # Skip problematic records
                            continue
                    
                    print(f"\nStats for {warc_file.name}:")
                    print(f"  - Records processed: {records_processed}")
                    print(f"  - Valid examples: {records_valid}")
                    if records_processed > 0:
                        print(f"  - Success rate: {records_valid/records_processed*100:.1f}%")
            
            except Exception as e:
                print(f"⚠ Error processing {warc_file.name}: {e}")
                print("Continuing with next WARC file...")
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ Extraction complete!")
        print(f"Total positive examples extracted: {len(positive_examples)}")
        print(f"{'='*60}\n")
        
        return positive_examples
    
    def create_negative_examples(self, num_examples: int = 40000) -> List[str]:
        """
        Create negative examples. 
        
        Option 1: If you have Common Crawl data, sample from there.
        Option 2: Generate synthetic low-quality text.
        Option 3: Use a pre-existing low-quality dataset.
        
        This is a placeholder - you'll need to adapt based on available data.
        """
        print("Creating negative examples...")
        
        # PLACEHOLDER: You need to provide your own negative examples
        # Here are some strategies:
        
        negative_examples = []
        
        # Strategy 1: If you have Common Crawl WARC files
        # cc_warc_path = "/path/to/commoncrawl.warc.gz"
        # Extract random pages similar to positive examples
        
        # Strategy 2: Generate synthetic low-quality text
        # This is a simple example - real negative examples would be better
        low_quality_patterns = [
            "Click here! Buy now! Free shipping! Order today! Call 1-800-" * 50,
            "Lorem ipsum " * 100,
            "404 Not Found. The page you requested could not be found. " * 50,
            "Copyright © All rights reserved. " * 100,
            "a " * 500,  # Very short repeated words
            "🔥🔥🔥 AMAZING DEAL 🔥🔥🔥 " * 50,
        ]
        
        for pattern in low_quality_patterns:
            for _ in range(num_examples // len(low_quality_patterns)):
                negative_examples.append(pattern + str(random.randint(1, 1000)))
        
        print(f"Created {len(negative_examples)} negative examples")
        print("WARNING: Using synthetic negative examples. For best results, use real Common Crawl data.")
        
        return negative_examples
    
    def prepare_training_data(self, positive_examples: List[str], 
                            negative_examples: List[str]) -> None:
        """
        Prepare training data in fastText format.
        Format: __label__high text...
               __label__low text...
        """
        print("Preparing training data...")
        
        with open(self.training_data_path, 'w', encoding='utf-8') as f:
            # Write positive examples
            for text in tqdm(positive_examples, desc="Writing positive examples"):
                # Clean text for fastText (remove newlines, etc.)
                clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                if clean_text:
                    f.write(f"__label__high {clean_text}\n")
            
            # Write negative examples
            for text in tqdm(negative_examples, desc="Writing negative examples"):
                clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
                if clean_text:
                    f.write(f"__label__low {clean_text}\n")
        
        print(f"Training data saved to {self.training_data_path}")
    
    def train_classifier(self, lr: float = 0.5, epoch: int = 25, 
                        wordNgrams: int = 2, dim: int = 100) -> None:
        """
        Train fastText classifier.
        """
        print("Training quality classifier...")
        
        model = fasttext.train_supervised(
            input=str(self.training_data_path),
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            dim=dim,
            loss='softmax',
            verbose=2
        )
        
        # Save model
        model.save_model(str(self.model_path))
        print(f"Model saved to {self.model_path}")
        
        # Test the model
        result = model.test(str(self.training_data_path))
        print(f"\nTraining Results:")
        print(f"Number of examples: {result[0]}")
        print(f"Precision: {result[1]:.4f}")
        print(f"Recall: {result[2]:.4f}")
        
        return model
    
    def train_pipeline(self, sample_size: int = 50000, 
                  max_positive: int = 40000,
                  max_negative: int = 40000,
                  skip_scraping: bool = False,
                  batch_size: int = 5000) -> None:  # Add batch_size parameter
        """
        Run the complete training pipeline.
        
        Args:
            sample_size: Number of URLs to sample from Wikipedia
            max_positive: Maximum number of positive examples to extract
            max_negative: Maximum number of negative examples to create
            skip_scraping: If True, skip URL scraping (use existing WARC)
            batch_size: Number of URLs to process in each batch (default: 5000)
        """
        # Step 1: Subsample URLs
        if not self.sampled_urls_path.exists():
            self.subsample_wikipedia_urls(sample_size)
        else:
            print(f"Using existing sampled URLs at {self.sampled_urls_path}")
        
        # Step 2: Scrape URLs in batches (optional - can skip if WARC exists)
        if not skip_scraping:
            # Check if any WARC files exist
            existing_warcs = list(self.base_dir.glob("positive_urls*.warc.gz"))
            if not existing_warcs:
                self.scrape_urls_in_batches(batch_size=batch_size)  # Use batched scraping
            else:
                print(f"Using existing WARC file(s) at {self.base_dir}")
        
        # Step 3: Extract positive examples
        positive_examples = self.extract_positive_examples(max_positive)
        
        if len(positive_examples) == 0:
            print("❌ Error: No positive examples extracted!")
            print("Check if WARC files exist and contain valid data.")
            return
        
        # Step 4: Create negative examples
        negative_examples = self.create_negative_examples(max_negative)
        
        # Step 5: Prepare training data
        self.prepare_training_data(positive_examples, negative_examples)
        
        # Step 6: Train classifier
        self.train_classifier()
        
        print("\n" + "="*60)
        print("✓ Training complete!")
        print(f"Model location: {self.model_path}")
        print("="*60)


def main():
    """
    Main function to train the quality classifier.
    """
def main():
    """
    Main function to train the quality classifier.
    """
    base_dir = "/Users/thefoolgy/Desktop/assignment1-basics-main/assignment4-data-main/cs336_data"
    
    trainer = QualityClassifierTrainer(base_dir)
    
    # Run the training pipeline with batch processing
    trainer.train_pipeline(
        sample_size=50000,
        max_positive=40000,
        max_negative=40000,
        skip_scraping=False,
        batch_size=5000  # Process 5000 URLs at a time
    )


if __name__ == "__main__":
    main()