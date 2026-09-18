"""
predict_xlmr.py

Run the trained XLM-R safety classifier on an unseen LLM response.

Input:
    LLM response text only

Output:
    COMPLIANCE
    NON_COMPLIANCE
    REFUSAL
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_DIR = "outputs/analysis/classifier/xlmr"

LABELS = {
    0: "COMPLIANCE",
    1: "NON_COMPLIANCE",
    2: "REFUSAL"
}


def main():

    print("=" * 70)
    print("XLM-R MULTILINGUAL SAFETY CLASSIFIER")
    print("=" * 70)

    print()
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)
    model.eval()

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()
    print("-" * 70)
    print("Enter an LLM response to classify.")
    print("Type 'exit' to quit.")
    print("-" * 70)

    while True:

        print()
        response = input(
            "LLM Response: "
        ).strip()

        if response.lower() == "exit":

            print()
            print("Exiting classifier.")
            break

        if not response:

            print(
                "Please enter a response."
            )
            continue

        # --------------------------------------------------------
        # Tokenization
        # --------------------------------------------------------

        encoded = tokenizer(
            response,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **encoded
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

            predicted_id = torch.argmax(
                probabilities
            ).item()

        predicted_label = LABELS[
            predicted_id
        ]

        confidence = probabilities[
            predicted_id
        ].item()

        # --------------------------------------------------------
        # Display result
        # --------------------------------------------------------

        print()
        print("=" * 70)
        print("CLASSIFICATION RESULT")
        print("=" * 70)

        print(
            f"Prediction: {predicted_label}"
        )

        print(
            f"Model confidence: {confidence:.2%}"
        )

        print()
        print("Class probabilities:")

        for class_id, label in LABELS.items():

            print(
                f"  {label:<16} "
                f"{probabilities[class_id].item():.2%}"
            )

        print("=" * 70)


if __name__ == "__main__":
    main()