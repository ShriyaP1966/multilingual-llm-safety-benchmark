from pathlib import Path

data = Path("data/working_dataset.csv").read_bytes()

bad_bytes = {0x91, 0x92, 0x93, 0x94}

for i, b in enumerate(data):
    if b in bad_bytes:
        start = max(0, i - 80)
        end = min(len(data), i + 80)

        print("=" * 60)
        print(f"Position: {i}")
        print(data[start:end])
        print("=" * 60)