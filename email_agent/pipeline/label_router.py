from email_agent.schemas import JobLabel

PROCESSED_LABEL = "PROCESSED"

def label_for_job(label: JobLabel) -> str:
    # Gmail label should match exactly what you want in your inbox
    return label.value