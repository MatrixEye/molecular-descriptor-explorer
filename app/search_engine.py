"""
search_engine.py

Purpose
-------
This module provides the main molecular search logic for the
Molecular Descriptor Explorer.

It is responsible for:

1. Accepting either a ChEMBL identifier or a SMILES string.
2. Validating and normalizing user input.
3. Canonicalizing SMILES with RDKit.
4. Performing exact database lookups.
5. Retrieving metadata, FMO descriptors, and FDA descriptors
   through DescriptorLoader.
6. Finding the most similar known molecules when an exact
   SMILES match is unavailable.
7. Calculating Morgan-fingerprint Tanimoto similarity.
8. Returning consistent structured dictionaries that can later
   be displayed by Streamlit or another web interface.

Performance
-----------
The precomputed fingerprints are collected once when SearchEngine
is initialized.

Similarity scores are calculated with RDKit's
BulkTanimotoSimilarity, which is considerably more efficient than
calling TanimotoSimilarity separately inside a Python loop.

Important
---------
The query fingerprint settings must match the settings used when
the stored fingerprints were generated.

The defaults below are:

    Morgan radius: 2
    Fingerprint size: 2048 bits

If metadata_with_fp.pkl was created using different settings,
change MORGAN_RADIUS and DEFAULT_FP_SIZE accordingly.
"""

import heapq
import re

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class SearchEngine:
    """
    Search molecular records by ChEMBL ID or SMILES and perform
    nearest-neighbour similarity searching.
    """

    MORGAN_RADIUS = 2
    DEFAULT_FP_SIZE = 2048
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 100

    # Supports identifiers such as:
    # CHEMBL25
    # CHEMBL25_1
    CHEMBL_PATTERN = re.compile(
        r"^CHEMBL\d+(?:_C\d+)?$",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        loader,
        fingerprint_radius=MORGAN_RADIUS,
        fingerprint_size=None,
    ):
        """
        Parameters
        ----------
        loader : DescriptorLoader
            Initialized DescriptorLoader containing molecular records
            and precomputed fingerprints.

        fingerprint_radius : int, optional
            Morgan fingerprint radius. This must match the radius used
            to generate the stored fingerprints.

        fingerprint_size : int or None, optional
            Number of fingerprint bits.

            When None, the size is inferred from the first stored
            fingerprint when possible. Otherwise, 2048 is used.
        """

        self.loader = loader

        if not isinstance(fingerprint_radius, int):
            raise TypeError("fingerprint_radius must be an integer.")

        if fingerprint_radius < 0:
            raise ValueError("fingerprint_radius cannot be negative.")

        self.fingerprint_radius = fingerprint_radius

        # Prepare fingerprint IDs and fingerprint objects once.
        self._prepare_fingerprints()

        if fingerprint_size is None:
            self.fingerprint_size = self._infer_fingerprint_size()
        else:
            if not isinstance(fingerprint_size, int):
                raise TypeError("fingerprint_size must be an integer.")

            if fingerprint_size <= 0:
                raise ValueError("fingerprint_size must be greater than zero.")

            self.fingerprint_size = fingerprint_size

        # Create the fingerprint generator once rather than once per query.
        self.fingerprint_generator = AllChem.GetMorganGenerator(
            radius=self.fingerprint_radius,
            fpSize=self.fingerprint_size,
        )

    # =============================================================
    # STARTUP PREPARATION
    # =============================================================

    def _prepare_fingerprints(self):
        """
        Store fingerprint identifiers and objects in parallel tuples.

        This avoids repeatedly iterating over and unpacking the loader's
        fingerprint dictionary for every similarity query.
        """

        fingerprint_items = [
            (str(chembl_id), fingerprint)
            for chembl_id, fingerprint in self.loader.fp_index.items()
            if fingerprint is not None
        ]

        self.fingerprint_ids = tuple(
            chembl_id
            for chembl_id, _ in fingerprint_items
        )

        self.fingerprints = tuple(
            fingerprint
            for _, fingerprint in fingerprint_items
        )

    def _infer_fingerprint_size(self):
        """
        Infer the fingerprint length from the stored fingerprints.

        Returns the default size when there are no stored fingerprints
        or the fingerprint object does not expose GetNumBits().
        """

        if not self.fingerprints:
            return self.DEFAULT_FP_SIZE

        first_fingerprint = self.fingerprints[0]

        if hasattr(first_fingerprint, "GetNumBits"):
            number_of_bits = first_fingerprint.GetNumBits()

            if number_of_bits > 0:
                return number_of_bits

        return self.DEFAULT_FP_SIZE

    # =============================================================
    # INPUT NORMALIZATION AND VALIDATION
    # =============================================================

    @staticmethod
    def _clean_input(user_input):
        """
        Remove surrounding whitespace from user input.

        Returns None if the input is not a non-empty string.
        """

        if not isinstance(user_input, str):
            return None

        cleaned_input = user_input.strip()

        if not cleaned_input:
            return None

        return cleaned_input

    @classmethod
    def _normalize_chembl_id(cls, chembl_id):
        """
        Normalize a ChEMBL identifier.

        ChEMBL identifiers are converted to uppercase and surrounding
        whitespace is removed.

        Returns None when the input does not resemble a supported
        ChEMBL identifier.
        """

        cleaned_id = cls._clean_input(chembl_id)

        if cleaned_id is None:
            return None

        if cls.CHEMBL_PATTERN.fullmatch(cleaned_id) is None:
            return None

        return cleaned_id.upper()

    def _canonicalize_smiles(self, smiles):
        """
        Validate and canonicalize a SMILES string.

        DescriptorLoader performs the central canonicalization so exact
        lookup and similarity search use the same representation.

        Returns
        -------
        str or None
            Canonical SMILES, or None for invalid input.
        """

        return self.loader.canonicalize_smiles(smiles)

    def _is_valid_smiles(self, smiles):
        """
        Return True when the input is a valid SMILES string.
        """

        return self._canonicalize_smiles(smiles) is not None

    def detect_input_type(self, user_input):
        """
        Determine whether input is a ChEMBL ID, SMILES, or invalid.

        Returns
        -------
        str
            "chembl_id", "smiles", or "invalid"
        """

        cleaned_input = self._clean_input(user_input)

        if cleaned_input is None:
            return "invalid"

        if self._normalize_chembl_id(cleaned_input) is not None:
            return "chembl_id"

        if self._is_valid_smiles(cleaned_input):
            return "smiles"

        return "invalid"

    # =============================================================
    # QUERY-FINGERPRINT CREATION
    # =============================================================

    def _create_query_fingerprint(self, canonical_smiles):
        """
        Create a Morgan fingerprint for a canonical SMILES string.
        """

        molecule = Chem.MolFromSmiles(canonical_smiles)

        if molecule is None:
            return None

        return self.fingerprint_generator.GetFingerprint(molecule)

    # =============================================================
    # EXACT ChEMBL ID LOOKUP
    # =============================================================

    def query_by_chembl_id(self, chembl_id):
        """
        Retrieve a complete molecular record using a ChEMBL ID.
        """

        normalized_id = self._normalize_chembl_id(chembl_id)

        if normalized_id is None:
            return {
                "status": "error",
                "query_type": "chembl_id",
                "query": chembl_id,
                "exact_match": False,
                "message": "Invalid ChEMBL identifier.",
                "result": None,
                "similar_molecules": [],
            }

        if not self.loader.exists(normalized_id):
            return {
                "status": "not_found",
                "query_type": "chembl_id",
                "query": normalized_id,
                "chembl_id": normalized_id,
                "exact_match": False,
                "message": (
                    "This ChEMBL identifier is not present "
                    "in the reference dataset."
                ),
                "result": None,
                "similar_molecules": [],
            }

        record = self.loader.get_full_record(normalized_id)

        return {
            "status": "success",
            "query_type": "chembl_id",
            "query": normalized_id,
            "chembl_id": normalized_id,
            "canonical_smiles": (
                record.get("smiles") if record is not None else None
            ),
            "exact_match": True,
            "data_source": "reference",
            "message": "Exact molecule found in the reference dataset.",
            "result": record,
            "similar_molecules": [],
        }

    # =============================================================
    # EXACT SMILES LOOKUP
    # =============================================================

    def query_by_smiles(self, smiles):
        """
        Search for an exact molecular structure using SMILES.

        The input is canonicalized before lookup. Therefore, alternative
        valid SMILES representations of the same molecule can match the
        canonical SMILES stored in the dataset.
        """

        cleaned_smiles = self._clean_input(smiles)

        if cleaned_smiles is None:
            return {
                "status": "error",
                "query_type": "smiles",
                "query": smiles,
                "exact_match": False,
                "message": "A non-empty SMILES string is required.",
                "result": None,
                "similar_molecules": [],
            }

        canonical_smiles = self._canonicalize_smiles(cleaned_smiles)

        if canonical_smiles is None:
            return {
                "status": "error",
                "query_type": "smiles",
                "query": cleaned_smiles,
                "exact_match": False,
                "message": "Invalid SMILES string.",
                "result": None,
                "similar_molecules": [],
            }

        chembl_id = self.loader.get_chembl_id_from_smiles(
            canonical_smiles
        )

        if chembl_id is None:
            return {
                "status": "not_found",
                "query_type": "smiles",
                "query": cleaned_smiles,
                "canonical_smiles": canonical_smiles,
                "exact_match": False,
                "message": (
                    "This molecule is not present in the "
                    "reference dataset."
                ),
                "result": None,
                "similar_molecules": [],
            }

        record = self.loader.get_full_record(chembl_id)

        return {
            "status": "success",
            "query_type": "smiles",
            "query": cleaned_smiles,
            "canonical_smiles": canonical_smiles,
            "chembl_id": chembl_id,
            "exact_match": True,
            "data_source": "reference",
            "message": "Exact molecule found in the reference dataset.",
            "result": record,
            "similar_molecules": [],
        }

    # =============================================================
    # SIMILARITY SEARCH
    # =============================================================

    def find_similar(self, smiles, top_k=DEFAULT_TOP_K):
        """
        Find the most similar molecules in the reference dataset.

        Similarity is calculated using Morgan fingerprints and the
        Tanimoto coefficient.

        Parameters
        ----------
        smiles : str
            Query molecule represented as a SMILES string.

        top_k : int, optional
            Number of nearest molecules to return.

        Returns
        -------
        dict
            Structured result containing similar molecules, similarity
            scores, and their reference descriptor records.
        """

        cleaned_smiles = self._clean_input(smiles)

        if cleaned_smiles is None:
            return {
                "status": "error",
                "query_type": "similarity",
                "query": smiles,
                "exact_match": False,
                "message": "A non-empty SMILES string is required.",
                "result": None,
                "similar_molecules": [],
            }

        canonical_smiles = self._canonicalize_smiles(cleaned_smiles)

        if canonical_smiles is None:
            return {
                "status": "error",
                "query_type": "similarity",
                "query": cleaned_smiles,
                "exact_match": False,
                "message": "Invalid SMILES string.",
                "result": None,
                "similar_molecules": [],
            }

        if not isinstance(top_k, int):
            return {
                "status": "error",
                "query_type": "similarity",
                "query": cleaned_smiles,
                "canonical_smiles": canonical_smiles,
                "exact_match": False,
                "message": "top_k must be an integer.",
                "result": None,
                "similar_molecules": [],
            }

        if top_k < 1:
            return {
                "status": "error",
                "query_type": "similarity",
                "query": cleaned_smiles,
                "canonical_smiles": canonical_smiles,
                "exact_match": False,
                "message": "top_k must be at least 1.",
                "result": None,
                "similar_molecules": [],
            }

        # Prevent a web request from asking the application to return
        # an unnecessarily large result.
        top_k = min(top_k, self.MAX_TOP_K)

        if not self.fingerprints:
            return {
                "status": "error",
                "query_type": "similarity",
                "query": cleaned_smiles,
                "canonical_smiles": canonical_smiles,
                "exact_match": False,
                "message": (
                    "No precomputed fingerprints are available "
                    "for similarity searching."
                ),
                "result": None,
                "similar_molecules": [],
            }

        query_fingerprint = self._create_query_fingerprint(
            canonical_smiles
        )

        if query_fingerprint is None:
            return {
                "status": "error",
                "query_type": "similarity",
                "query": cleaned_smiles,
                "canonical_smiles": canonical_smiles,
                "exact_match": False,
                "message": "Could not generate a query fingerprint.",
                "result": None,
                "similar_molecules": [],
            }

        try:
            similarity_scores = DataStructs.BulkTanimotoSimilarity(
                query_fingerprint,
                self.fingerprints,
            )

        except Exception as error:
            return {
                "status": "error",
                "query_type": "similarity",
                "query": cleaned_smiles,
                "canonical_smiles": canonical_smiles,
                "exact_match": False,
                "message": (
                    "Similarity calculation failed. Check that the "
                    "stored fingerprints use the same RDKit fingerprint "
                    "type and size as the query fingerprints."
                ),
                "details": str(error),
                "result": None,
                "similar_molecules": [],
            }

        number_of_hits = min(top_k, len(similarity_scores))

        # nlargest avoids sorting every result when only a few of the
        # highest-scoring molecules are required.
        top_indices = heapq.nlargest(
            number_of_hits,
            range(len(similarity_scores)),
            key=similarity_scores.__getitem__,
        )

        similar_molecules = []

        for index in top_indices:
            chembl_id = self.fingerprint_ids[index]
            similarity = float(similarity_scores[index])
            record = self.loader.get_full_record(chembl_id)

            similar_molecules.append(
                {
                    "chembl_id": chembl_id,
                    "similarity": similarity,
                    "data_source": "nearest_neighbour",
                    "result": record,
                }
            )

        return {
            "status": "success",
            "query_type": "similarity",
            "query": cleaned_smiles,
            "canonical_smiles": canonical_smiles,
            "exact_match": False,
            "data_source": "nearest_neighbour",
            "message": (
                "The query molecule was not found exactly. "
                "The nearest molecules from the reference dataset "
                "are shown instead."
            ),
            "result": None,
            "similar_molecules": similar_molecules,
        }

    # =============================================================
    # UNIFIED PUBLIC QUERY METHOD
    # =============================================================

    def query(self, user_input, top_k=DEFAULT_TOP_K):
        """
        Main public search method used by the website.

        Behaviour
        ---------
        ChEMBL ID:
            Return the exact record when available.

        SMILES:
            First attempt an exact canonical-SMILES lookup.
            If the molecule is absent, return the nearest molecules.

        Invalid input:
            Return a structured error.
        """

        cleaned_input = self._clean_input(user_input)

        if cleaned_input is None:
            return {
                "status": "error",
                "query_type": "invalid",
                "query": user_input,
                "exact_match": False,
                "message": (
                    "Enter a ChEMBL identifier or a valid SMILES string."
                ),
                "result": None,
                "similar_molecules": [],
            }

        input_type = self.detect_input_type(cleaned_input)

        if input_type == "chembl_id":
            return self.query_by_chembl_id(cleaned_input)

        if input_type == "smiles":
            exact_result = self.query_by_smiles(cleaned_input)

            if exact_result["status"] == "not_found":
                return self.find_similar(
                    cleaned_input,
                    top_k=top_k,
                )

            return exact_result

        return {
            "status": "error",
            "query_type": "invalid",
            "query": cleaned_input,
            "exact_match": False,
            "message": (
                "Invalid input. Enter a ChEMBL identifier such as "
                "'CHEMBL25' or a valid SMILES string."
            ),
            "result": None,
            "similar_molecules": [],
        }

    # =============================================================
    # ENGINE INFORMATION
    # =============================================================

    def summary(self):
        """
        Return basic search-engine configuration information.
        """

        return {
            "searchable_fingerprints": len(self.fingerprints),
            "fingerprint_radius": self.fingerprint_radius,
            "fingerprint_size": self.fingerprint_size,
            "default_top_k": self.DEFAULT_TOP_K,
            "maximum_top_k": self.MAX_TOP_K,
        }
