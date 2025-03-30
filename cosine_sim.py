import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration options
# qt_type = "FB_OE"
qt_type = "FB"
# qt_type = "OE"

# target_file = f"{qt_type}_CA"
target_file = f"{qt_type}_DB"
# target_file = f"{qt_type}_DS"

# Base directories (using os.path.join for cross-platform compatibility)
BASE_INPUT_DIR = os.path.join("data", "input", "ans_b4_exp")
BASE_OUTPUT_DIR = os.path.join("data", "output", "similarity", "ans_b4_exp")

# Function to perform enhanced text cleaning
def enhanced_clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle numbers with commas (e.g., "65,535" -> "65535")
    text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
    
    # Remove all punctuation and replace with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common conjunctions that don't affect meaning
    text = re.sub(r'\band\b', ' ', text)
    
    return text

# Function to specifically normalize and compare answers
def normalized_compare(text1, text2):
    """Special comparison for handling specific formats"""
    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0
    
    # Normalize both texts
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    
    # Handle different number formats (with/without commas)
    if re.match(r'^\d+$', t1.replace(',', '')) and re.match(r'^\d+$', t2.replace(',', '')):
        return 1 if t1.replace(',', '') == t2.replace(',', '') else 0
    
    # Handle differently formatted lists
    # E.g., "a, b, and c" vs "a,b,c"
    t1_tokens = set(re.sub(r'[^\w]', ' ', t1).split())
    t2_tokens = set(re.sub(r'[^\w]', ' ', t2).split())
    t1_tokens = {t for t in t1_tokens if t and not t.lower() in ['and', 'or', 'the', 'a', 'an']}
    t2_tokens = {t for t in t2_tokens if t and not t.lower() in ['and', 'or', 'the', 'a', 'an']}
    
    if t1_tokens and t2_tokens:
        # If all tokens match, return 1
        if t1_tokens == t2_tokens:
            return 1
    
    return 0

# Token-based similarity function using Jaccard similarity
def token_based_similarity(text1, text2):
    # First check if there's an exact match after normalization
    exact_match = normalized_compare(text1, text2)
    if exact_match == 1:
        return 1
    
    # Clean texts
    clean_text1 = enhanced_clean_text(text1)
    clean_text2 = enhanced_clean_text(text2)
    
    # Tokenize
    tokens1 = set(clean_text1.split())
    tokens2 = set(clean_text2.split())
    
    # Remove common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'to', 'for'}
    tokens1 = tokens1 - stop_words
    tokens2 = tokens2 - stop_words
    
    # Calculate Jaccard similarity
    if not tokens1 or not tokens2:
        return 0
    
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    if union == 0:
        return 0
    
    return intersection / union

# Function to calculate both cosine and token-based similarities
def calculate_similarities(df):
    # TF-IDF based cosine similarity
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    
    # Apply enhanced cleaning to both columns
    df['Correct_Answer_Clean'] = df['Correct Answer'].apply(enhanced_clean_text)
    df['LLM_Answer_Clean'] = df['LLM_Answer'].apply(enhanced_clean_text)
    
    # Create TF-IDF vectors
    all_texts = list(df['Correct_Answer_Clean']) + list(df['LLM_Answer_Clean'])
    
    # Check for empty strings and handle them
    if all(text == "" for text in all_texts):
        # Return zeros if all texts are empty
        return df, [0] * len(df), [0] * len(df), [0] * len(df)
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix back into correct answers and LLM answers
    correct_vectors = tfidf_matrix[:len(df)]
    llm_vectors = tfidf_matrix[len(df):]
    
    # Calculate cosine similarity for each pair
    cosine_similarities = []
    for i in range(len(df)):
        if correct_vectors[i].nnz == 0 or llm_vectors[i].nnz == 0:
            similarity = 0
        else:
            similarity = cosine_similarity(correct_vectors[i], llm_vectors[i])[0][0]
        cosine_similarities.append(similarity)
    
    # Calculate token-based similarity for each pair
    token_similarities = []
    for i in range(len(df)):
        similarity = token_based_similarity(
            df.iloc[i]['Correct Answer'], 
            df.iloc[i]['LLM_Answer']
        )
        token_similarities.append(similarity)
    
    # Add similarities to dataframe
    df['Cosine_Similarity'] = cosine_similarities
    df['Token_Similarity'] = token_similarities
    
    # Use the maximum of both similarity methods
    df['Combined_Similarity'] = df[['Cosine_Similarity', 'Token_Similarity']].max(axis=1)
    
    return df, cosine_similarities, token_similarities, df['Combined_Similarity'].tolist()

# Function to process a single LLM model
def process_llm(llm_name):
    input_dir = os.path.join(BASE_INPUT_DIR, target_file)
    
    # Construct the file path
    file_path = os.path.join(input_dir, f"{llm_name}_{qt_type}.csv")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found for {llm_name}: {file_path}")
        return None
    
    print(f"\n{'='*50}")
    print(f"Analyzing responses from LLM: {llm_name}")
    print(f"{'='*50}")
    
    # Load the CSV file
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
    except Exception as e:
        print(f"Error reading file for {llm_name}: {e}")
        return None
    
    # Calculate similarities
    df, cosine_similarities, token_similarities, combined_similarities = calculate_similarities(df)
    
    # Calculate overall statistics for combined similarity
    avg_similarity = np.mean(combined_similarities)
    median_similarity = np.median(combined_similarities)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot cosine similarity scores
    plt.subplot(2, 2, 1)
    plt.bar(range(len(cosine_similarities)), cosine_similarities, color='skyblue')
    plt.axhline(y=np.mean(cosine_similarities), color='r', linestyle='-', 
                label=f'Average: {np.mean(cosine_similarities):.3f}')
    plt.xlabel('Question Index')
    plt.ylabel('Cosine Similarity')
    plt.title('TF-IDF Cosine Similarity')
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot token-based similarity scores
    plt.subplot(2, 2, 2)
    plt.bar(range(len(token_similarities)), token_similarities, color='lightgreen')
    plt.axhline(y=np.mean(token_similarities), color='r', linestyle='-', 
                label=f'Average: {np.mean(token_similarities):.3f}')
    plt.xlabel('Question Index')
    plt.ylabel('Token Similarity')
    plt.title('Token-based Similarity')
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot combined similarity scores
    plt.subplot(2, 2, 3)
    plt.bar(range(len(combined_similarities)), combined_similarities, color='lightsalmon')
    plt.axhline(y=avg_similarity, color='r', linestyle='-', 
                label=f'Average: {avg_similarity:.3f}')
    plt.xlabel('Question Index')
    plt.ylabel('Combined Similarity')
    plt.title(f'Combined Similarity Between Correct and {llm_name} Answers')
    plt.ylim(0, 1)
    plt.legend()
    
    # Plot histogram of combined similarities
    plt.subplot(2, 2, 4)
    plt.hist(combined_similarities, bins=10, color='lightsalmon', edgecolor='black')
    plt.axvline(x=avg_similarity, color='r', linestyle='-', 
                label=f'Average: {avg_similarity:.3f}')
    plt.xlabel('Combined Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Combined Similarity Scores for {llm_name}')
    plt.ylim(0, len(df)/2)
    plt.legend()
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {llm_name} (Combined Similarity):")
    print(f"Average similarity: {avg_similarity:.3f}")
    print(f"Median similarity: {median_similarity:.3f}")
    print(f"Minimum similarity: {min(combined_similarities):.3f}")
    print(f"Maximum similarity: {max(combined_similarities):.3f}")
    
    # Find questions with highest and lowest similarity
    highest_idx = np.argmax(combined_similarities)
    lowest_idx = np.argmin(combined_similarities)
    
    print(f"\nHighest similarity question for {llm_name}:")
    print(f"Question: {df.iloc[highest_idx]['Question']}")
    print(f"Correct Answer: {df.iloc[highest_idx]['Correct Answer']}")
    print(f"LLM Answer: {df.iloc[highest_idx]['LLM_Answer']}")
    print(f"Combined Similarity: {combined_similarities[highest_idx]:.3f}")
    print(f"Cosine Similarity: {cosine_similarities[highest_idx]:.3f}")
    print(f"Token Similarity: {token_similarities[highest_idx]:.3f}")
    
    print(f"\nLowest similarity question for {llm_name}:")
    print(f"Question: {df.iloc[lowest_idx]['Question']}")
    print(f"Correct Answer: {df.iloc[lowest_idx]['Correct Answer']}")
    print(f"LLM Answer: {df.iloc[lowest_idx]['LLM_Answer']}")
    print(f"Combined Similarity: {combined_similarities[lowest_idx]:.3f}")
    print(f"Cosine Similarity: {cosine_similarities[lowest_idx]:.3f}")
    print(f"Token Similarity: {token_similarities[lowest_idx]:.3f}")
    
    # Export results to CSV
    df_export = df[['Question', 'Correct Answer', 'LLM_Answer', 
                    'Cosine_Similarity', 'Token_Similarity', 'Combined_Similarity']]
    df_export = df_export.sort_values('Combined_Similarity', ascending=False)
    
    # Create output directories if they don't exist
    output_dir = os.path.join(BASE_OUTPUT_DIR, target_file)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'enhanced_similarity_results_{llm_name}.csv')
    df_export.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Save the visualization
    viz_file = os.path.join(output_dir, f'enhanced_similarity_analysis_{llm_name}.png')
    plt.savefig(viz_file)
    print(f"Visualization saved to {viz_file}")
    
    # Display top and bottom 5 results
    print(f"\nTop 5 most similar answers for {llm_name}:")
    print(df_export.head(5))
    
    print(f"\nBottom 5 least similar answers for {llm_name}:")
    print(df_export.tail(5))
    
    # Check for specific problematic examples - numbers with commas
    number_with_commas = df[df['Correct Answer'].str.contains('\d+,\d+', regex=True, na=False) | 
                           df['LLM_Answer'].str.contains('\d+,\d+', regex=True, na=False)]
    if not number_with_commas.empty:
        print("\nChecking examples with numbers that have commas:")
        for i, row in number_with_commas.iterrows():
            print(f"Question: {row['Question']}")
            print(f"Correct Answer: {row['Correct Answer']}")
            print(f"LLM Answer: {row['LLM_Answer']}")
            print(f"Combined Similarity: {row['Combined_Similarity']:.3f}")
            print(f"Cosine Similarity: {row['Cosine_Similarity']:.3f}")
            print(f"Token Similarity: {row['Token_Similarity']:.3f}")
            print("---")
    
    plt.close()  # Close the plot to free memory
    
    return {
        'llm_name': llm_name,
        'avg_similarity': avg_similarity,
        'median_similarity': median_similarity,
        'min_similarity': min(combined_similarities),
        'max_similarity': max(combined_similarities)
    }

# List of all LLM models to process
llm_list = [
    "chatgpt-4o-latest",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "deepseek-v3-fw",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite",
    "GPT-3.5-Turbo",
    "GPT-4o-Mini",
    "grok-2",
    "llama-3.3-70b",
    "mistral-medium",
    "o1",
    "o1-mini",
    "o3-mini-high"
]

# Process all LLMs and collect results
results = []
for llm in llm_list:
    result = process_llm(llm)
    if result:
        results.append(result)

# Create a summary comparison of all LLMs
if results:
    # Convert results to DataFrame for easy visualization
    summary_df = pd.DataFrame(results)
    
    # Sort by average similarity
    summary_df = summary_df.sort_values('avg_similarity', ascending=False)
    
    # Create summary output directory if it doesn't exist
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Save summary to CSV
    summary_file = os.path.join(BASE_OUTPUT_DIR, "llm_similarity_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n{'='*50}")
    print(f"Summary of all LLM performance saved to {summary_file}")
    
    # Create summary comparison plot
    plt.figure(figsize=(14, 8))
    
    # Plot average similarities
    plt.bar(summary_df['llm_name'], summary_df['avg_similarity'], color='lightblue')
    
    # Add value labels on top of each bar
    for i, v in enumerate(summary_df['avg_similarity']):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.xlabel('LLM Model')
    plt.ylabel('Average Similarity Score')
    plt.title('Comparison of LLM Performance by Average Similarity Score with answers before explanation')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_dir = os.path.join(BASE_OUTPUT_DIR, target_file)
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_file = os.path.join(comparison_dir, "llm_comparison.png")
    plt.savefig(comparison_file)
    print(f"Comparison visualization saved to {comparison_file}")
    
    # Print summary table
    print("\nLLM Performance Summary (sorted by average similarity):")
    print(summary_df[['llm_name', 'avg_similarity', 'median_similarity', 'min_similarity', 'max_similarity']])