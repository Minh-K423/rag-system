from xmlrpc import client

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from setup.eda import generate_point, data_availability

from pathlib import Path
import os

from config import Settings
config = Settings()


def main():
    qdrant_client = QdrantClient(host="localhost", port=6333)

    npy_dir_path = config.NPY_DIR
    json_dir_path = config.JSON_DIR

    qdrant_client.recreate_collection(
        collection_name="video_features_collection",
        vectors_config=VectorParams(size=512, distance=Distance.DOT),
    )

    npy_dir = Path(npy_dir_path)
    json_dir = Path(json_dir_path)

    if not npy_dir.exists() or not json_dir.exists():
        print("[-] One or both of the specified directories do not exist.")
        return

    for npy_file in npy_dir.glob("*.npy"):
        if not npy_file.is_file():
            continue

        json_file = json_dir / (npy_file.stem + ".json")
        if not json_file.is_file():
            print(f"[-] Corresponding JSON file not found for: {npy_file.name}")
            continue

        if data_availability(npy_file, json_file):
            points = generate_point(npy_file, json_file)
            print(f"\n[+] Generated {len(points)} points for video: {npy_file.stem}")

            # Upload points to Qdrant
            qdrant_client.upsert(
                collection_name="video_features_collection",
                points=[point for point in points],
            )
            print(f"[+] Uploaded points for video: {npy_file.stem} to Qdrant.")
        else:
            print(f"[-] Data not available for video: {npy_file.stem}")


if __name__ == "__main__":
    main()