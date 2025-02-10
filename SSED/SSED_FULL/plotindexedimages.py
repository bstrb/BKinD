import matplotlib.pyplot as plt

# Provided data lines
lines = [
"165 images processed, 90 hits (54.5%), 1 indexable (1.1% of hits, 0.6% overall), 1 crystals, 32.5 images/sec.",
"228 images processed, 152 hits (66.7%), 1 indexable (0.7% of hits, 0.4% overall), 1 crystals, 12.5 images/sec.",
"291 images processed, 213 hits (73.2%), 4 indexable (1.9% of hits, 1.4% overall), 4 crystals, 12.5 images/sec.",
"329 images processed, 249 hits (75.7%), 5 indexable (2.0% of hits, 1.5% overall), 5 crystals, 7.5 images/sec.",
"371 images processed, 291 hits (78.4%), 8 indexable (2.7% of hits, 2.2% overall), 8 crystals, 8.2 images/sec.",
"473 images processed, 393 hits (83.1%), 9 indexable (2.3% of hits, 1.9% overall), 9 crystals, 20.3 images/sec.",
"524 images processed, 444 hits (84.7%), 9 indexable (2.0% of hits, 1.7% overall), 9 crystals, 10.2 images/sec.",
"575 images processed, 493 hits (85.7%), 9 indexable (1.8% of hits, 1.6% overall), 9 crystals, 10.1 images/sec.",
"619 images processed, 536 hits (86.6%), 9 indexable (1.7% of hits, 1.5% overall), 9 crystals, 8.6 images/sec.",
"665 images processed, 581 hits (87.4%), 12 indexable (2.1% of hits, 1.8% overall), 12 crystals, 9.2 images/sec.",
"729 images processed, 644 hits (88.3%), 12 indexable (1.9% of hits, 1.6% overall), 12 crystals, 12.7 images/sec.",
"864 images processed, 778 hits (90.0%), 12 indexable (1.5% of hits, 1.4% overall), 12 crystals, 26.8 images/sec.",
]

# Lists to store parsed data
images_processed_list = []
indexable_list = []

# Parse each line to extract counts
for line in lines:
    parts = line.split(',')
    
    # Extract "images processed" count
    images_processed = int(parts[0].split()[0])
    
    # Extract "indexable" count
    indexable = int(parts[2].split()[0])
    
    # Store in lists
    images_processed_list.append(images_processed)
    indexable_list.append(indexable)

# Compute the differences in indexable counts between successive lines
new_indexed = [indexable_list[0]]  # for the first entry, all indexable are new
for i in range(1, len(indexable_list)):
    new_indexed_count = indexable_list[i] - indexable_list[i - 1]
    new_indexed.append(new_indexed_count)

# Print the results: new indexable vs images processed
for imgs, new_idx in zip(images_processed_list, new_indexed):
    print(f"At {imgs} images processed, new indexable images: {new_idx}")

# Plotting the new indexable images against images processed
plt.figure(figsize=(8, 6))
plt.plot(images_processed_list, new_indexed, marker='o', linestyle='-', color='b', label='New Indexable Images')
plt.xlabel('Images Processed')
plt.ylabel('New Indexable Images')
plt.title('New Indexable Images vs. Images Processed')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()