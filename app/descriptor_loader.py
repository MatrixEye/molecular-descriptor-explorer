"""
descriptor_loader.py

Purpose
-------
This module loads and provides fast access to the molecular descriptor
datasets used by the Molecular Descriptor Explorer.

It is responsible for:

1. Loading molecular metadata, including precomputed RDKit fingerprints.
2. Loading global FMO descriptors.
3. Loading global FDA descriptors.
4. Building fast lookup indexes using CHEMBLID_Conf.
5. Looking up molecules by ChEMBL ID.
6. Looking up molecules by SMILES.
7. Canonicalizing user-entered SMILES with RDKit before searching.
8. Returning a complete molecular record containing:
       - ChEMBL ID
       - canonical SMILES
       - metadata
       - FMO descriptors
       - FDA descriptors
9. Providing access to precomputed fingerprints for similarity searching.

The class is designed so that the datasets are loaded only once when the
application starts. The website can then perform repeated fast lookups
without re-reading the files from disk.

Future ML prediction code should remain separate from this module.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem


class DescriptorLoader:
    """
    Load molecular metadata and descriptor datasets and create
    fast lookup structures for the web application.
    """

    REQUIRED_ID_COLUMN = "CHEMBLID_Conf"
    REQUIRED_SMILES_COLUMN = "smiles"
    FINGERPRINT_COLUMN = "fingerprint"

    def __init__(self, metadata_path, fmo_path, fda_path):
        """
        Parameters
        ----------
        metadata_path : str or Path
            Path to metadata file.

            This will normally be metadata_with_fp.pkl because the
            precomputed molecular fingerprints are stored there.

        fmo_path : str or Path
            Path to global_FMO.csv.

        fda_path : str or Path
            Path to global_FDA.csv.
        """

        self.metadata_path = Path(metadata_path)
        self.fmo_path = Path(fmo_path)
        self.fda_path = Path(fda_path)

        # ---------------------------------------------------------
        # Check that required files actually exist
        # ---------------------------------------------------------
        self._validate_files()

        # ---------------------------------------------------------
        # Load datasets ONCE when the application starts
        # ---------------------------------------------------------
        self.metadata = self._load_metadata(self.metadata_path)
        self.fmo = self._load_csv(self.fmo_path)
        self.fda = self._load_csv(self.fda_path)

        # ---------------------------------------------------------
        # Validate required columns
        # ---------------------------------------------------------
        self._validate_columns()

        # ---------------------------------------------------------
        # Build fast lookup indexes
        # ---------------------------------------------------------
        self._build_indexes()

    # =============================================================
    # DATA LOADING
    # =============================================================

    def _validate_files(self):
        """
        Check that all required data files exist.
        """

        paths = [
            self.metadata_path,
            self.fmo_path,
            self.fda_path,
        ]

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required data file was not found: {path}"
                )

    @staticmethod
    def _load_metadata(path):
        """
        Load metadata.

        Supports both pickle and CSV files so the storage format
        can be changed later without rewriting the rest of the class.
        """

        suffix = path.suffix.lower()

        if suffix in {".pkl", ".pickle"}:
            dataframe = pd.read_pickle(path)

        elif suffix == ".csv":
            dataframe = pd.read_csv(path)

        else:
            raise ValueError(
                f"Unsupported metadata format: {suffix}. "
                "Use .pkl, .pickle, or .csv."
            )

        return DescriptorLoader._clean_dataframe(dataframe)

    @staticmethod
    def _load_csv(path):
        """
        Load a CSV descriptor table.
        """

        dataframe = pd.read_csv(path)

        return DescriptorLoader._clean_dataframe(dataframe)

    @staticmethod
    def _clean_dataframe(dataframe):
        """
        Remove accidental index columns such as 'Unnamed: 0'
        that are commonly created when pandas DataFrames are
        written to CSV.
        """

        unnamed_columns = [
            column
            for column in dataframe.columns
            if str(column).startswith("Unnamed:")
        ]

        if unnamed_columns:
            dataframe = dataframe.drop(columns=unnamed_columns)

        return dataframe

    # =============================================================
    # VALIDATION
    # =============================================================

    def _validate_columns(self):
        """
        Make sure the datasets contain the columns needed by
        the application.
        """

        metadata_required = {
            self.REQUIRED_ID_COLUMN,
            self.REQUIRED_SMILES_COLUMN,
        }

        missing_metadata = metadata_required - set(self.metadata.columns)

        if missing_metadata:
            raise ValueError(
                "Metadata file is missing required columns: "
                f"{sorted(missing_metadata)}"
            )

        if self.REQUIRED_ID_COLUMN not in self.fmo.columns:
            raise ValueError(
                f"FMO dataset must contain "
                f"'{self.REQUIRED_ID_COLUMN}'."
            )

        if self.REQUIRED_ID_COLUMN not in self.fda.columns:
            raise ValueError(
                f"FDA dataset must contain "
                f"'{self.REQUIRED_ID_COLUMN}'."
            )

    # =============================================================
    # INDEX CREATION
    # =============================================================

    def _build_indexes(self):
        """
        Build lookup indexes for fast access.

        CHEMBLID_Conf is used as the primary identifier.

        SMILES are also indexed so that an exact molecular structure
        can be found after the user-entered SMILES is canonicalized.
        """

        # Convert identifiers to strings for consistent lookup
        self.metadata[self.REQUIRED_ID_COLUMN] = (
            self.metadata[self.REQUIRED_ID_COLUMN].astype(str)
        )

        self.fmo[self.REQUIRED_ID_COLUMN] = (
            self.fmo[self.REQUIRED_ID_COLUMN].astype(str)
        )

        self.fda[self.REQUIRED_ID_COLUMN] = (
            self.fda[self.REQUIRED_ID_COLUMN].astype(str)
        )

        # Remove repeated records for the same conformer ID.
        # The duplicates differ only in an accidental saved index column.
        self.metadata = self.metadata.drop_duplicates(
            subset=[self.REQUIRED_ID_COLUMN],
            keep="first",
        ).copy()

        # ---------------------------------------------------------
        # Main indexed tables
        # ---------------------------------------------------------

        self.metadata_index = self.metadata.set_index(
            self.REQUIRED_ID_COLUMN,
            drop=True,
        )
        self.fmo_index = self.fmo.set_index(
            self.REQUIRED_ID_COLUMN,
            drop=True,
        )

        self.fda_index = self.fda.set_index(
            self.REQUIRED_ID_COLUMN,
            drop=True,
        )

        # ---------------------------------------------------------
        # Check whether primary IDs are unique
        # ---------------------------------------------------------

        if not self.metadata_index.index.is_unique:
            raise ValueError(
                "CHEMBLID_Conf is not unique in the metadata dataset."
            )

        # ---------------------------------------------------------
        # Fast canonical-SMILES -> ChEMBL lookup
        # ---------------------------------------------------------

        valid_smiles = self.metadata_index[
            self.REQUIRED_SMILES_COLUMN
        ].dropna()

        self.smiles_to_chembl = dict(
            zip(
                valid_smiles.astype(str),
                valid_smiles.index,
            )
        )

        # ---------------------------------------------------------
        # Fast ChEMBL -> fingerprint lookup
        # ---------------------------------------------------------

        if self.FINGERPRINT_COLUMN in self.metadata_index.columns:

            fingerprints = self.metadata_index[
                self.FINGERPRINT_COLUMN
            ].dropna()

            self.fp_index = fingerprints.to_dict()

        else:
            self.fp_index = {}

    # =============================================================
    # SMILES UTILITIES
    # =============================================================

    @staticmethod
    def canonicalize_smiles(smiles):
        """
        Validate and canonicalize a SMILES string using RDKit.

        Parameters
        ----------
        smiles : str
            User-provided SMILES.

        Returns
        -------
        str or None
            Canonical RDKit SMILES if valid.
            None if the SMILES cannot be parsed.
        """

        if not isinstance(smiles, str):
            return None

        smiles = smiles.strip()

        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True,
        )

    # =============================================================
    # BASIC LOOKUP FUNCTIONS
    # =============================================================

    def exists(self, chembl_id):
        """
        Check whether a ChEMBL identifier exists.
        """

        chembl_id = str(chembl_id)

        return chembl_id in self.metadata_index.index

    def smiles_exists(self, smiles):
        """
        Check whether a molecule exists in the dataset
        using a SMILES query.
        """

        canonical_smiles = self.canonicalize_smiles(smiles)

        if canonical_smiles is None:
            return False

        return canonical_smiles in self.smiles_to_chembl

    def get_chembl_id_from_smiles(self, smiles):
        """
        Convert a user-provided SMILES to canonical SMILES
        and return the corresponding CHEMBLID_Conf.

        Returns None if the molecule is not present.
        """

        canonical_smiles = self.canonicalize_smiles(smiles)

        if canonical_smiles is None:
            return None

        return self.smiles_to_chembl.get(canonical_smiles)

    # =============================================================
    # DESCRIPTOR ACCESS
    # =============================================================

    def get_metadata(self, chembl_id):
        """
        Return metadata for a given CHEMBLID_Conf.
        """

        chembl_id = str(chembl_id)

        try:
            return self.metadata_index.loc[chembl_id].to_dict()

        except KeyError:
            return None

    def get_fmo(self, chembl_id):
        """
        Return global FMO descriptors for a given CHEMBLID_Conf.
        """

        chembl_id = str(chembl_id)

        try:
            return self.fmo_index.loc[chembl_id].to_dict()

        except KeyError:
            return None

    def get_fda(self, chembl_id):
        """
        Return global FDA descriptors for a given CHEMBLID_Conf.
        """

        chembl_id = str(chembl_id)

        try:
            return self.fda_index.loc[chembl_id].to_dict()

        except KeyError:
            return None

    # =============================================================
    # FINGERPRINT ACCESS
    # =============================================================

    def get_fingerprint(self, chembl_id):
        """
        Return the precomputed fingerprint for a molecule.

        Used by search_engine.py for similarity searches.
        """

        chembl_id = str(chembl_id)

        return self.fp_index.get(chembl_id)

    # =============================================================
    # COMPLETE MOLECULE RECORD
    # =============================================================

    def get_full_record(self, chembl_id):
        """
        Return all available information for a molecule.

        Returns
        -------
        dict or None

        Example
        -------
        {
            "chembl_id": "CHEMBL123_1",
            "smiles": "...",
            "metadata": {...},
            "FMO": {...},
            "FDA": {...}
        }
        """

        chembl_id = str(chembl_id)

        metadata = self.get_metadata(chembl_id)

        if metadata is None:
            return None

        return {
            "chembl_id": chembl_id,
            "smiles": metadata.get(self.REQUIRED_SMILES_COLUMN),
            "metadata": metadata,
            "FMO": self.get_fmo(chembl_id),
            "FDA": self.get_fda(chembl_id),
        }

    def get_full_record_from_smiles(self, smiles):
        """
        Return the complete database record for a SMILES query.

        The input SMILES is first canonicalized with RDKit.

        Returns None when:
        - the SMILES is invalid, or
        - the molecule is not present in the dataset.

        An unseen molecule should subsequently be passed to
        search_engine.py for nearest-neighbour searching.
        """

        chembl_id = self.get_chembl_id_from_smiles(smiles)

        if chembl_id is None:
            return None

        return self.get_full_record(chembl_id)

    # =============================================================
    # DATASET INFORMATION
    # =============================================================

    def get_all_chembl_ids(self):
        """
        Return all ChEMBL identifiers.

        A tuple is returned rather than recreating a mutable list
        from the underlying index.
        """

        return tuple(self.metadata_index.index)

    def __len__(self):
        """
        Number of molecules/records in the metadata dataset.
        """

        return len(self.metadata_index)

    def summary(self):
        """
        Return basic information useful for application startup
        checks and debugging.
        """

        return {
            "metadata_records": len(self.metadata_index),
            "fmo_records": len(self.fmo_index),
            "fda_records": len(self.fda_index),
            "fingerprints": len(self.fp_index),
            "unique_smiles": len(self.smiles_to_chembl),
        }
