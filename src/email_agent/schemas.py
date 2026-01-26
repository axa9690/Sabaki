from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class JobLabel(str, Enum):
    APPLIED = "APPLIED"
    ASSESSMENTS = "ASSESSMENTS"
    IN_PROCESS = "IN PROCESS"        
    INTERVIEWS = "INTERVIEWS"
    REJECTED = "REJECTED"
    OTP_SECURITY = "OTP_SECURITY"   
    RECOMMENDATIONS = "RECOMMENDATIONS"
    JOB_ALERTS = "JOB_ALERTS"
    ADVERTISEMENTS = "ADVERTISEMENTS"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EmailAnalysis(BaseModel):
    label: JobLabel = Field(..., description="Job pipeline label to apply")
    urgency: Urgency = Field(..., description="How time-sensitive this email is")
    needs_reply: bool = Field(..., description="Should Anand reply to this email?")
    reasoning_brief: str = Field(..., description="One short line why this label was chosen")
