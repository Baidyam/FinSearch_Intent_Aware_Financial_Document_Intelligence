import os

# Root dataset directory (gdown downloads folder contents directly into Dataset/)
DATASET_DIR = os.path.join(".", "Dataset")

# ── Knowledge Base (Document Corpus) ──────────────────────────────────────────
KNOWLEDGE_BASE_DIR              = os.path.join(DATASET_DIR, "Knowledge base")

CONSUMER_PROTECTION_PATH        = os.path.join(KNOWLEDGE_BASE_DIR, "Consumer_Protection")
PAYMENT_INDUSTRY_PATH           = os.path.join(KNOWLEDGE_BASE_DIR, "Payment_Industry")
REGULATORY_PATH                 = os.path.join(KNOWLEDGE_BASE_DIR, "Regulatory")

SEC_FILINGS_DIR                 = os.path.join(KNOWLEDGE_BASE_DIR, "sec_filings")
EDGAR_PATH                      = os.path.join(SEC_FILINGS_DIR, "EDGAR")
SEC_METADATA_PATH               = os.path.join(SEC_FILINGS_DIR, "metadata")
SEC_QA_DATASETS_PATH            = os.path.join(SEC_FILINGS_DIR, "QA_Datasets")

# Individual SEC QA files
FINANCEBENCH_PATH               = os.path.join(SEC_QA_DATASETS_PATH, "financebench_train.json")
FINQA_CORPUS_PATH               = os.path.join(SEC_QA_DATASETS_PATH, "finqa_corpus.json")
FINQA_QUERIES_PATH              = os.path.join(SEC_QA_DATASETS_PATH, "finqa_queries.json")

# ── Query & Evaluation Datasets ───────────────────────────────────────────────
QUERY_DATASET_DIR               = os.path.join(DATASET_DIR, "Query_Dataset")

BANKING77_PATH                  = os.path.join(QUERY_DATASET_DIR, "Banking77")
BANKING77_TRAIN                 = os.path.join(BANKING77_PATH, "train.csv")
BANKING77_TEST                  = os.path.join(BANKING77_PATH, "test.csv")

FIQA_PATH                       = os.path.join(QUERY_DATASET_DIR, "FiQA")
FIQA_CORPUS                     = os.path.join(FIQA_PATH, "corpus.csv")
FIQA_QUERIES                    = os.path.join(FIQA_PATH, "queries.csv")
FIQA_QRELS_TRAIN                = os.path.join(FIQA_PATH, "qrels_train.csv")
FIQA_QRELS_DEV                  = os.path.join(FIQA_PATH, "qrels_dev.csv")
FIQA_QRELS_TEST                 = os.path.join(FIQA_PATH, "qrels_test.csv")

IEEE_FRAUD_PATH                 = os.path.join(QUERY_DATASET_DIR, "IEEE-CIS Fraud")
IEEE_FRAUD_IDENTITY             = os.path.join(IEEE_FRAUD_PATH, "train_identity.csv.zip")
IEEE_FRAUD_TRANSACTION          = os.path.join(IEEE_FRAUD_PATH, "train_transaction.csv.zip")

# ── Synthetic Data ─────────────────────────────────────────────────────────────
SYNTHETIC_DIR                   = os.path.join(DATASET_DIR, "Synthetic")

CFPB_COMPLAINTS_PATH            = os.path.join(SYNTHETIC_DIR, "CFPB Consumer Complaints", "complaints.csv.zip")
COMPLAINT_PROCEDURES_PATH       = os.path.join(SYNTHETIC_DIR, "customer_complaint_handling_procedure")
DISPUTES_PATH                   = os.path.join(SYNTHETIC_DIR, "Disputes", "disputes.csv")
TRANSACTIONS_PATH               = os.path.join(SYNTHETIC_DIR, "Transactions", "transactions.csv")
INTENT_FILTERS_PATH             = os.path.join(SYNTHETIC_DIR, "Transactions", "intent_filters.json")
USERS_DB_PATH                   = os.path.join(SYNTHETIC_DIR, "Users", "users.db")


def verify_paths():
    """Check all dataset paths exist. Run this after setup_data.py."""
    all_paths = {
        # Knowledge Base
        "Consumer Protection"           : CONSUMER_PROTECTION_PATH,
        "Payment Industry"              : PAYMENT_INDUSTRY_PATH,
        "Regulatory"                    : REGULATORY_PATH,
        "EDGAR SEC Filings"             : EDGAR_PATH,
        "SEC Metadata"                  : SEC_METADATA_PATH,
        "FinanceBench"                  : FINANCEBENCH_PATH,
        "FinQA Corpus"                  : FINQA_CORPUS_PATH,
        "FinQA Queries"                 : FINQA_QUERIES_PATH,
        # Query Datasets
        "Banking77 Train"               : BANKING77_TRAIN,
        "Banking77 Test"                : BANKING77_TEST,
        "FiQA Corpus"                   : FIQA_CORPUS,
        "FiQA Queries"                  : FIQA_QUERIES,
        "FiQA Qrels (train)"            : FIQA_QRELS_TRAIN,
        "FiQA Qrels (dev)"              : FIQA_QRELS_DEV,
        "FiQA Qrels (test)"             : FIQA_QRELS_TEST,
        "IEEE Fraud Identity"           : IEEE_FRAUD_IDENTITY,
        "IEEE Fraud Transaction"        : IEEE_FRAUD_TRANSACTION,
        # Synthetic
        "CFPB Complaints"               : CFPB_COMPLAINTS_PATH,
        "Complaint Procedures"          : COMPLAINT_PROCEDURES_PATH,
        "Disputes"                      : DISPUTES_PATH,
        "Transactions"                  : TRANSACTIONS_PATH,
        "Intent Filters"                : INTENT_FILTERS_PATH,
        "Users DB"                      : USERS_DB_PATH,
    }

    print("Verifying dataset paths...\n")
    all_ok = True
    for name, path in all_paths.items():
        status = "OK     " if os.path.exists(path) else "MISSING"
        if status.strip() == "MISSING":
            all_ok = False
        print(f"  [{status}] {name}: {path}")

    print()
    if all_ok:
        print("All paths verified successfully!")
    else:
        print("Some paths are missing. Run: python setup_data.py")


if __name__ == "__main__":
    verify_paths()
