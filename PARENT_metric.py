import json


def read_file_lines(file_path):
    """Reads all lines from a file and returns a list of lines."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


def parse_rdf_table(table_line):
    """Parses a single line representing an RDF table into a structured format."""
    # Assuming RDF table is represented as a list of tuples in the line
    return json.loads(table_line)


def calculate_overlap(ngram, rdf_data):
    """Calculates overlap between an n-gram and RDF data."""
    tokens = set(ngram)
    rdf_tokens = set(token for entry in rdf_data for token in entry)
    overlap = len(tokens.intersection(rdf_tokens))
    return overlap / len(tokens) if tokens else 0


def compute_metrics(predictions, references, rdf_tables):
    """Computes precision, recall, and F-score based on predictions, references, and RDF tables."""
    precision_sum, recall_sum, f_score_sum = 0.0, 0.0, 0.0

    for pred, refs, rdf in zip(predictions, references, rdf_tables):
        rdf_data = parse_rdf_table(rdf)
        pred_tokens = pred.split()

        # Precision: Proportion of predicted tokens that overlap with RDF data
        precision = calculate_overlap(pred_tokens, rdf_data)

        # Recall: Average overlap of reference tokens with RDF data
        recall_list = [calculate_overlap(ref.split(), rdf_data) for ref in refs.split('\t')]
        recall = sum(recall_list) / len(recall_list) if recall_list else 0

        # F-score: Harmonic mean of precision and recall
        f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_sum += precision
        recall_sum += recall
        f_score_sum += f_score

    num_items = len(predictions)
    avg_precision = precision_sum / num_items
    avg_recall = recall_sum / num_items
    avg_f_score = f_score_sum / num_items

    return avg_precision, avg_recall, avg_f_score


# Example usage
references_file = "path/to/references.txt"
generations_file = "path/to/generations.txt"
tables_file = "path/to/tables.txt"

references = read_file_lines(references_file)
generations = read_file_lines(generations_file)
rdf_tables = read_file_lines(tables_file)

avg_precision, avg_recall, avg_f_score = compute_metrics(generations, references, rdf_tables)
print(f"Precision: {avg_precision}, Recall: {avg_recall}, F-score: {avg_f_score}")
