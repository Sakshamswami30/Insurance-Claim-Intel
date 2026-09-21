from src.doc_generator_messy import generate_messy_batch

generate_messy_batch(
    count=30,
    output_dir="data/samples/claims_messy",
    truth_path="data/samples/claims_ground_truth_messy.json",
)