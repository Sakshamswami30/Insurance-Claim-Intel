from src.doc_generator import generate_batch

generate_batch(
    count=200,
    output_dir="data/samples/claims",
    truth_path="data/samples/claims_ground_truth.json",
)