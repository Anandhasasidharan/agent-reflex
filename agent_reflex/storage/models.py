from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    agent_id = Column(String(255), nullable=False)
    task_description = Column(Text, default="")
    success = Column(Integer, default=0)
    reliability_score = Column(Float, default=0.0)
    failure_type = Column(String(100), nullable=True)
    cause_node_id = Column(String(255), nullable=True)
    causal_responsibility_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    steps = relationship("TraceStepRecord", back_populates="session", cascade="all, delete-orphan")
    edges = relationship("GraphEdgeRecord", back_populates="session", cascade="all, delete-orphan")


class TraceStepRecord(Base):
    __tablename__ = "trace_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id"), nullable=False)
    node_id = Column(String(255), nullable=False)
    agent_id = Column(String(255), nullable=False)
    step_index = Column(Integer, nullable=False)
    observation = Column(Text, default="")
    thought = Column(Text, default="")
    action = Column(String(255), default="")
    result = Column(Text, default="")
    parent_id = Column(String(255), nullable=True)
    subtask_id = Column(String(255), nullable=True)
    execution_time_ms = Column(Float, default=0.0)
    error_flag = Column(Integer, default=0)

    session = relationship("SessionRecord", back_populates="steps")


class GraphEdgeRecord(Base):
    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id"), nullable=False)
    source_id = Column(String(255), nullable=False)
    target_id = Column(String(255), nullable=False)
    edge_type = Column(String(50), default="control_flow")

    session = relationship("SessionRecord", back_populates="edges")


class RecoveryLogRecord(Base):
    __tablename__ = "recovery_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False)
    agent_id = Column(String(255), default="")
    playbook_name = Column(String(255), nullable=False)
    selector = Column(String(50), default="adaptive")
    success = Column(Integer, default=0)
    partial = Column(Integer, default=0)
    recovery_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ReliabilityRecord(Base):
    __tablename__ = "reliability_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False)
    score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
