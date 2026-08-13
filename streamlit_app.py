"""
streamlit_app.py

Web interface for the Molecular Descriptor Explorer.

The application:
- searches by ChEMBL conformer ID or SMILES;
- displays exact reference data;
- performs nearest-neighbour similarity searches;
- provides 2D and interactive 3D molecular views;
- reserves space for future ML predictions.
"""

import hmac
import html
import math
import re
from numbers import Integral, Real

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors

from app.config import FDA_FILE, FMO_FILE, METADATA_FP_FILE
from app.descriptor_loader import DescriptorLoader
from app.search_engine import SearchEngine


# =============================================================
# PAGE CONFIGURATION
# =============================================================

st.set_page_config(
    page_title="Molecular Descriptor Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# TEMPORARY PASSWORD PROTECTION
# =============================================================

def check_password():
    """Check the submitted password without exposing the secret."""

    entered_password = st.session_state.get(
        "password_input",
        "",
    )

    expected_password = str(
        st.secrets["auth"]["password"]
    )

    if hmac.compare_digest(
        entered_password,
        expected_password,
    ):
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False

    st.session_state["password_input"] = ""


def require_password():
    """Stop the application until the correct password is entered."""

    try:
        auth_settings = st.secrets["auth"]
        password_enabled = bool(
            auth_settings.get(
                "enabled",
                True,
            )
        )
        auth_settings["password"]

    except (KeyError, FileNotFoundError):
        st.error(
            "Application password has not been configured."
        )
        st.stop()

    if not password_enabled:
        return

    if st.session_state.get("authenticated", False):
        with st.sidebar:
            if st.button(
                "Log out",
                use_container_width=True,
            ):
                st.session_state["authenticated"] = False
                st.rerun()

        return

    st.title("Molecular Descriptor Explorer")
    st.write(
        "This application is currently in private testing."
    )

    st.text_input(
        "Password",
        type="password",
        key="password_input",
        on_change=check_password,
        placeholder="Enter the testing password",
    )

    if (
        "authenticated" in st.session_state
        and not st.session_state["authenticated"]
    ):
        st.error("Incorrect password.")

    st.stop()


require_password()

# =============================================================
# STYLE
# =============================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(
                    circle at 8% 4%,
                    rgba(20, 184, 166, 0.08),
                    transparent 24%
                ),
                #f7f9fc;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }

        .hero {
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.4rem;
            border-radius: 22px;
            color: white;
            background:
                linear-gradient(
                    120deg,
                    #0f172a 0%,
                    #123c56 55%,
                    #0f766e 100%
                );
            box-shadow: 0 16px 38px rgba(15, 23, 42, 0.16);
        }

        .hero h1 {
            margin: 0;
            color: white;
            font-size: 2.35rem;
            letter-spacing: -0.04em;
        }

        .hero p {
            max-width: 850px;
            margin: 0.7rem 0 0;
            color: rgba(255, 255, 255, 0.84);
            font-size: 1.02rem;
        }

        .result-heading {
            margin-top: 1rem;
            margin-bottom: 0.35rem;
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 750;
            letter-spacing: -0.025em;
        }

        .reference-badge,
        .neighbour-badge {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .reference-badge {
            border: 1px solid #99f6e4;
            color: #115e59;
            background: #ccfbf1;
        }

        .neighbour-badge {
            border: 1px solid #bfdbfe;
            color: #1e40af;
            background: #dbeafe;
        }

        .notice-card {
            padding: 1rem 1.2rem;
            margin: 0.7rem 0 1.3rem;
            border: 1px solid #dbe4ee;
            border-left: 4px solid #0f766e;
            border-radius: 12px;
            background: white;
            color: #334155;
        }

        .ml-card {
            padding: 1.3rem;
            border: 1px dashed #94a3b8;
            border-radius: 16px;
            background: #f8fafc;
            color: #334155;
        }

        .identity-panel {
            padding: 1rem;
            margin-top: 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
        }

        .property-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.9rem;
        }

        .property-card {
            min-width: 0;
            min-height: 88px;
            padding: 0.85rem 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: white;
            box-shadow: 0 5px 15px rgba(15, 23, 42, 0.04);
        }

        .property-label {
            margin-bottom: 0.38rem;
            color: #64748b;
            font-size: 0.75rem;
            font-weight: 650;
        }

        .property-value,
        .formula-value {
            color: #0f172a;
            font-size: 1.28rem;
            font-weight: 720;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .formula-value sub {
            position: relative;
            bottom: -0.15em;
            font-size: 0.65em;
            line-height: 0;
        }

        .property-unit {
            display: block;
            margin-top: 0.15rem;
            color: #64748b;
            font-size: 0.7rem;
            font-weight: 500;
        }

        div[data-testid="stMetric"] {
            min-height: 100px;
            padding: 0.85rem 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: white;
            box-shadow: 0 5px 15px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stForm"] {
            padding: 1.2rem 1.3rem;
            border: 1px solid #dbe4ee;
            border-radius: 16px;
            background: white;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: white;
            border-radius: 16px;
        }

        .section-caption {
            margin-top: 0.4rem;
            color: #64748b;
            font-size: 0.78rem;
        }

        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 0.82rem;
            text-align: center;
        }

        @media (max-width: 900px) {
            .property-grid {
                grid-template-columns: 1fr;
            }

            .hero {
                padding: 1.5rem;
            }

            .hero h1 {
                font-size: 1.85rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# DATABASE
# =============================================================

@st.cache_resource(show_spinner="Loading molecular reference database...")
def load_search_engine():
    """Load the database and fingerprints only once."""

    loader = DescriptorLoader(
        metadata_path=METADATA_FP_FILE,
        fmo_path=FMO_FILE,
        fda_path=FDA_FILE,
    )

    return SearchEngine(loader)


# =============================================================
# MOLECULAR VISUALIZATION
# =============================================================

def create_2d_image(smiles, size=(700, 500)):
    """Create an RDKit 2D structure image."""

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return Draw.MolToImage(
        molecule,
        size=size,
        kekulize=True,
    )


@st.cache_data(show_spinner=False)
def create_3d_molblock(smiles):
    """Generate a reproducible 3D conformer from SMILES."""

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    molecule = Chem.AddHs(molecule)

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 61453
    parameters.enforceChirality = True

    embed_status = AllChem.EmbedMolecule(
        molecule,
        parameters,
    )

    if embed_status != 0:
        return None

    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(
                molecule,
                maxIters=300,
            )
        else:
            AllChem.UFFOptimizeMolecule(
                molecule,
                maxIters=300,
            )
    except Exception:
        pass

    return Chem.MolToMolBlock(molecule)


def render_2d_structure(smiles):
    """Display a 2D structure."""

    image = create_2d_image(smiles)

    if image is None:
        st.warning("The 2D structure could not be generated.")
        return

    st.image(
        image,
        use_container_width=True,
    )


def render_3d_structure(smiles, height=470):
    """Display an interactive 3D structure."""

    molblock = create_3d_molblock(smiles)

    if molblock is None:
        st.warning(
            "A 3D conformer could not be generated for this molecule."
        )
        return

    viewer = py3Dmol.view(
        width=620,
        height=height,
    )

    viewer.addModel(
        molblock,
        "mol",
    )

    viewer.setStyle(
        {
            "stick": {
                "radius": 0.16,
                "colorscheme": "Jmol",
            },
            "sphere": {
                "scale": 0.26,
                "colorscheme": "Jmol",
            },
        }
    )

    viewer.setBackgroundColor("#f8fafc")
    viewer.zoomTo()

    components.html(
        viewer._make_html(),
        height=height,
        scrolling=False,
    )

    st.caption(
        "Drag to rotate • Scroll to zoom • "
        "RDKit-generated conformer for visualization; "
        "not a QM-optimized geometry."
    )


def display_structure_viewer(smiles):
    """Display 2D and interactive 3D structure tabs."""

    view_2d, view_3d = st.tabs(
        [
            "2D structure",
            "Interactive 3D",
        ]
    )

    with view_2d:
        render_2d_structure(smiles)

    with view_3d:
        render_3d_structure(smiles)


# =============================================================
# VALUE HELPERS
# =============================================================

def is_missing(value):
    """Check whether a value is missing."""

    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_value(value):
    """Format values for display."""

    if is_missing(value):
        return "—"

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, Integral):
        return f"{int(value):,}"

    if isinstance(value, Real):
        number = float(value)

        if math.isnan(number):
            return "—"

        if abs(number) >= 10000:
            return f"{number:,.4f}"

        return f"{number:.5g}"

    return str(value)


def get_first_available(data, candidate_names):
    """Find a descriptor using case-insensitive column matching."""

    if not data:
        return None

    normalized_data = {
        str(key).strip().lower(): value
        for key, value in data.items()
    }

    for candidate in candidate_names:
        normalized_candidate = str(candidate).strip().lower()

        if normalized_candidate not in normalized_data:
            continue

        value = normalized_data[normalized_candidate]

        if not is_missing(value):
            return value

    return None


# =============================================================
# DESCRIPTOR TABLES
# =============================================================

def prepare_descriptor_table(data):
    """Convert a descriptor dictionary to a table."""

    if not data:
        return pd.DataFrame(
            columns=[
                "Property",
                "Value",
            ]
        )

    rows = []

    for property_name, value in data.items():
        property_name = str(property_name)

        if property_name == "fingerprint":
            continue

        if property_name.startswith("Unnamed:"):
            continue

        rows.append(
            {
                "Property": property_name,
                "Value": format_value(value),
            }
        )

    return pd.DataFrame(rows)


def display_descriptor_table(data):
    """Display a descriptor table."""

    table = prepare_descriptor_table(data)

    if table.empty:
        st.info("No data are available in this section.")
        return

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=min(
            620,
            38 + len(table) * 35,
        ),
    )


# =============================================================
# BASIC MOLECULAR PROPERTIES
# =============================================================

def calculate_basic_properties(smiles):
    """Calculate basic structural properties with RDKit."""

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return {}

    return {
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "molecular_weight": Descriptors.MolWt(molecule),
        "donors": rdMolDescriptors.CalcNumHBD(molecule),
        "acceptors": rdMolDescriptors.CalcNumHBA(molecule),
    }


def formula_to_html(formula):
    """Convert formula numbers into HTML subscripts."""

    safe_formula = html.escape(str(formula))

    return re.sub(
        r"(\d+)",
        r"<sub>\1</sub>",
        safe_formula,
    )


def display_basic_properties(smiles):
    """
    Display basic molecular properties.

    HTML is deliberately constructed without leading indentation so
    Streamlit does not interpret it as a Markdown code block.
    """

    properties = calculate_basic_properties(smiles)

    if not properties:
        return

    formula = formula_to_html(
        properties["formula"]
    )

    molecular_weight = (
        f"{properties['molecular_weight']:.2f}"
    )

    donors = int(properties["donors"])
    acceptors = int(properties["acceptors"])

    property_html = (
        '<div class="property-grid">'
        '<div class="property-card">'
        '<div class="property-label">Molecular formula</div>'
        f'<div class="formula-value">{formula}</div>'
        '</div>'
        '<div class="property-card">'
        '<div class="property-label">Molecular weight</div>'
        f'<div class="property-value">{molecular_weight}'
        '<span class="property-unit">g mol⁻¹</span>'
        '</div>'
        '</div>'
        '<div class="property-card">'
        '<div class="property-label">H-bond donors</div>'
        f'<div class="property-value">{donors}</div>'
        '</div>'
        '<div class="property-card">'
        '<div class="property-label">H-bond acceptors</div>'
        f'<div class="property-value">{acceptors}</div>'
        '</div>'
        '</div>'
    )

    st.markdown(
        property_html,
        unsafe_allow_html=True,
    )


# =============================================================
# KEY DESCRIPTOR SUMMARIES
# =============================================================

def display_summary_cards(title, data, specifications):
    """Display selected descriptors in rows of three cards."""

    if not data:
        st.subheader(title)
        st.info("No descriptor data are available.")
        return

    available_values = []

    for label, candidate_names in specifications:
        value = get_first_available(
            data,
            candidate_names,
        )

        if value is not None:
            available_values.append(
                (
                    label,
                    value,
                )
            )

    st.subheader(title)

    if not available_values:
        st.info("The selected descriptors are not available.")
        return

    for row_start in range(
        0,
        len(available_values),
        3,
    ):
        row_values = available_values[
            row_start:row_start + 3
        ]

        columns = st.columns(
            len(row_values)
        )

        for column, (label, value) in zip(
            columns,
            row_values,
        ):
            column.metric(
                label,
                format_value(value),
            )


def display_fmo_summary(fmo_data):
    """Display key FMO descriptors."""

    specifications = [
        (
            "HOMO",
            ["E_HOMO", "HOMO"],
        ),
        (
            "LUMO",
            ["E_LUMO", "LUMO"],
        ),
        (
            "Chemical potential (μ)",
            [
                "mu",
                "chemical_potential",
                "chemical potential",
            ],
        ),
        (
            "Electronegativity",
            [
                "el_neg",
                "electronegativity",
            ],
        ),
        (
            "Hardness (η)",
            [
                "eta",
                "hardness",
            ],
        ),
        (
            "Electrophilicity (ω)",
            [
                "omega",
                "electrophilicity",
            ],
        ),
    ]

    display_summary_cards(
        "Key FMO descriptors",
        fmo_data,
        specifications,
    )

    st.caption(
        "FMO values are displayed as stored in the reference dataset."
    )


def display_fda_summary(fda_data):
    """Display key FDA descriptors."""

    specifications = [
        (
            "Ionization potential (IP)",
            [
                "IP",
                "ionization_potential",
                "ionization potential",
            ],
        ),
        (
            "Electron affinity (EA)",
            [
                "EA",
                "electron_affinity",
                "electron affinity",
            ],
        ),
        (
            "Chemical potential (μ)",
            [
                "mu",
                "chemical_potential",
                "chemical potential",
            ],
        ),
        (
            "Hardness (η)",
            [
                "eta",
                "hardness",
            ],
        ),
        (
            "Softness",
            [
                "soft",
                "softness",
            ],
        ),
        (
            "Electrophilicity (ω)",
            [
                "omega",
                "electrophilicity",
            ],
        ),
    ]

    display_summary_cards(
        "Key FDA descriptors",
        fda_data,
        specifications,
    )

    st.caption(
        "FDA values are displayed as stored in the reference dataset."
    )


# =============================================================
# EXACT RESULT
# =============================================================

def display_exact_result(result):
    """Display an exact reference-dataset result."""

    record = result.get("result")

    if not record:
        st.error("The molecular record could not be retrieved.")
        return

    chembl_id = str(
        record.get(
            "chembl_id",
            "Unknown identifier",
        )
    )

    smiles = record.get("smiles")

    if not smiles:
        st.error("The record does not contain a SMILES string.")
        return

    st.markdown(
        (
            f'<div class="result-heading">'
            f'{html.escape(chembl_id)}'
            f'</div>'
            f'<span class="reference-badge">'
            f'REFERENCE DATA'
            f'</span>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="notice-card">'
            'Exact molecular structure found in the reference dataset. '
            'The displayed FMO and FDA values are stored reference '
            'values, not ML predictions.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns(
        [1, 1.2],
        gap="large",
    )

    # ---------------------------------------------------------
    # LEFT: structure and identity
    # ---------------------------------------------------------

    with left_column:
        st.subheader("Molecular structure")

        display_structure_viewer(smiles)

        st.subheader("Molecular identity")

        st.markdown("**Canonical SMILES**")

        st.code(
            smiles,
            language=None,
        )

        display_basic_properties(smiles)

    # ---------------------------------------------------------
    # RIGHT: key descriptors
    # ---------------------------------------------------------

    with right_column:
        display_fmo_summary(
            record.get("FMO")
        )

        st.write("")

        display_fda_summary(
            record.get("FDA")
        )

    st.divider()

    metadata_tab, fmo_tab, fda_tab, prediction_tab = st.tabs(
        [
            "Metadata",
            "Global FMO descriptors",
            "Global FDA descriptors",
            "ML prediction",
        ]
    )

    with metadata_tab:
        st.subheader("Molecular metadata")

        display_descriptor_table(
            record.get("metadata")
        )

    with fmo_tab:
        st.subheader("Global FMO descriptors")

        display_descriptor_table(
            record.get("FMO")
        )

    with fda_tab:
        st.subheader("Global FDA descriptors")

        display_descriptor_table(
            record.get("FDA")
        )

    with prediction_tab:
        st.markdown(
            (
                '<div class="ml-card">'
                '<strong>ML prediction module reserved</strong>'
                '<br><br>'
                'When the trained model becomes available, this section '
                'can compare reference values with model predictions.'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


# =============================================================
# SIMILARITY RESULTS
# =============================================================

def display_similarity_card(hit, position):
    """Display one nearest-neighbour card."""

    record = hit.get("result")

    if not record:
        st.warning("Reference record unavailable.")
        return

    chembl_id = str(
        hit.get(
            "chembl_id",
            "Unknown",
        )
    )

    similarity = float(
        hit.get(
            "similarity",
            0.0,
        )
    )

    smiles = record.get("smiles")

    st.markdown(
        f"#### {position}. {html.escape(chembl_id)}"
    )

    st.markdown(
        '<span class="neighbour-badge">NEAREST NEIGHBOUR</span>',
        unsafe_allow_html=True,
    )

    image = create_2d_image(
        smiles,
        size=(500, 340),
    )

    if image is not None:
        st.image(
            image,
            use_container_width=True,
        )

    st.metric(
        "Tanimoto similarity",
        f"{similarity:.3f}",
    )

    st.progress(
        min(
            max(similarity, 0.0),
            1.0,
        )
    )

    with st.expander("View reference descriptors"):
        section = st.selectbox(
            "Descriptor section",
            [
                "Metadata",
                "FMO",
                "FDA",
            ],
            key=f"section_{position}_{chembl_id}",
        )

        if section == "Metadata":
            display_descriptor_table(
                record.get("metadata")
            )

        elif section == "FMO":
            display_descriptor_table(
                record.get("FMO")
            )

        else:
            display_descriptor_table(
                record.get("FDA")
            )


def display_similarity_results(result):
    """Display nearest reference molecules for an unseen query."""

    canonical_smiles = result.get("canonical_smiles")

    st.markdown(
        (
            '<div class="result-heading">Query molecule</div>'
            '<span class="neighbour-badge">'
            'NOT IN REFERENCE DATASET'
            '</span>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="notice-card">'
            'No exact reference record was found. The following '
            'molecules are the closest structures available in the '
            'database. Their descriptors belong to those reference '
            'molecules.'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    query_structure, query_information = st.columns(
        [1, 1.2],
        gap="large",
    )

    with query_structure:
        st.subheader("Query structure")

        display_structure_viewer(
            canonical_smiles
        )

        st.subheader("Molecular identity")

        st.markdown("**Canonical SMILES**")

        st.code(
            canonical_smiles,
            language=None,
        )

        display_basic_properties(
            canonical_smiles
        )

    with query_information:
        st.subheader("ML prediction")

        st.markdown(
            (
                '<div class="ml-card">'
                '<strong>ML predictions are not available yet.</strong>'
                '<br><br>'
                'The future model will predict descriptor values for '
                'this unseen molecule and display applicability or '
                'uncertainty information here.'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    similar_molecules = result.get(
        "similar_molecules",
        [],
    )

    st.subheader("Nearest reference molecules")

    st.caption(
        "Ranked using Morgan-fingerprint Tanimoto similarity."
    )

    if not similar_molecules:
        st.info("No similar reference molecules were found.")
        return

    number_of_columns = 3

    for row_start in range(
        0,
        len(similar_molecules),
        number_of_columns,
    ):
        columns = st.columns(
            number_of_columns,
            gap="medium",
        )

        row_hits = similar_molecules[
            row_start:row_start + number_of_columns
        ]

        for offset, hit in enumerate(row_hits):
            position = row_start + offset + 1

            with columns[offset]:
                with st.container(border=True):
                    display_similarity_card(
                        hit,
                        position,
                    )


# =============================================================
# MAIN PAGE
# =============================================================

st.markdown(
    (
        '<div class="hero">'
        '<h1>Molecular Descriptor Explorer</h1>'
        '<p>'
        'Explore calculated molecular descriptors using exact '
        'structure lookup and RDKit-based molecular similarity search.'
        '</p>'
        '</div>'
    ),
    unsafe_allow_html=True,
)


try:
    engine = load_search_engine()

except Exception as error:
    st.error("The molecular reference database could not be loaded.")
    st.exception(error)
    st.stop()


if "search_result" not in st.session_state:
    st.session_state.search_result = None


with st.form("molecular_search_form"):
    search_column, number_column, button_column = st.columns(
        [5, 1.2, 1.2],
        vertical_alignment="bottom",
    )

    with search_column:
        user_input = st.text_input(
            "ChEMBL conformer ID or SMILES",
            placeholder=(
                "Example: CHEMBL435190_C02 "
                "or CC(=O)Oc1ccccc1C(=O)O"
            ),
        )

    with number_column:
        top_k = st.selectbox(
            "Neighbours",
            options=[
                3,
                5,
                6,
                9,
                10,
            ],
            index=1,
        )

    with button_column:
        submitted = st.form_submit_button(
            "Search",
            type="primary",
            use_container_width=True,
        )


if submitted:
    cleaned_input = user_input.strip()

    if not cleaned_input:
        st.session_state.search_result = None

        st.warning(
            "Enter a ChEMBL conformer identifier or valid SMILES."
        )

    else:
        with st.spinner("Searching the reference database..."):
            st.session_state.search_result = engine.query(
                cleaned_input,
                top_k=top_k,
            )


result = st.session_state.search_result


if result is not None:
    st.write("")

    if result.get("status") == "success":
        if result.get("exact_match"):
            display_exact_result(result)
        else:
            display_similarity_results(result)

    elif result.get("status") == "not_found":
        st.warning(
            result.get(
                "message",
                "No matching molecule was found.",
            )
        )

    else:
        st.error(
            result.get(
                "message",
                "The molecular search failed.",
            )
        )


# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:
    st.title("About")

    st.write(
        "Search calculated molecular descriptors using a "
        "ChEMBL conformer identifier or molecular SMILES."
    )

    st.divider()

    engine_summary = engine.summary()

    st.metric(
        "Searchable fingerprints",
        f"{engine_summary['searchable_fingerprints']:,}",
    )

    st.metric(
        "Morgan radius",
        engine_summary["fingerprint_radius"],
    )

    st.metric(
        "Fingerprint size",
        f"{engine_summary['fingerprint_size']} bits",
    )

    st.divider()

    st.markdown("### Data labels")

    st.markdown(
        '<span class="reference-badge">REFERENCE DATA</span>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Calculated values stored in the reference database."
    )

    st.markdown(
        '<span class="neighbour-badge">NEAREST NEIGHBOUR</span>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Values belong to a structurally similar reference molecule."
    )

    st.markdown("**ML prediction**")

    st.caption(
        "Reserved for the future trained prediction model."
    )


# =============================================================
# FOOTER
# =============================================================

st.markdown(
    (
        '<div class="footer">'
        'Molecular Descriptor Explorer · Reference data, molecular '
        'similarity search and future ML property prediction'
        '</div>'
    ),
    unsafe_allow_html=True,
)
