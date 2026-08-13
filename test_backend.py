from app.config import METADATA_FP_FILE, FMO_FILE, FDA_FILE
from app.descriptor_loader import DescriptorLoader
from app.search_engine import SearchEngine


loader = DescriptorLoader(
    metadata_path=METADATA_FP_FILE,
    fmo_path=FMO_FILE,
    fda_path=FDA_FILE,
)

engine = SearchEngine(loader)

print("Loader:", loader.summary())
print("Search engine:", engine.summary())

first_id = loader.get_all_chembl_ids()[0]
result = engine.query(first_id)

print("Test ID:", first_id)
print("Status:", result["status"])
