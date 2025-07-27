import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/optimization_db')

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class OptimizationProblem(Base):
    """Table to store optimization problem configurations."""
    __tablename__ = "optimization_problems"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Problem configuration (stored as JSON)
    products_data = Column(Text)  # JSON string
    resources_data = Column(Text)  # JSON string
    consumption_matrix = Column(Text)  # JSON string
    
    # Problem metadata
    num_products = Column(Integer)
    num_resources = Column(Integer)
    problem_type = Column(String(100), default="production_planning")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert problem to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'products_data': json.loads(self.products_data) if self.products_data else [],
            'resources_data': json.loads(self.resources_data) if self.resources_data else [],
            'consumption_matrix': json.loads(self.consumption_matrix) if self.consumption_matrix else [],
            'num_products': self.num_products,
            'num_resources': self.num_resources,
            'problem_type': self.problem_type
        }

class OptimizationSolution(Base):
    """Table to store optimization solutions."""
    __tablename__ = "optimization_solutions"
    
    id = Column(Integer, primary_key=True, index=True)
    problem_id = Column(Integer, nullable=False)  # Foreign key to OptimizationProblem
    solved_at = Column(DateTime, default=datetime.utcnow)
    
    # Solution data
    status = Column(String(50))
    objective_value = Column(Float)
    variables = Column(Text)  # JSON string of variable values
    shadow_prices = Column(Text)  # JSON string
    slack_variables = Column(Text)  # JSON string
    
    # Solution metadata
    solver_time_seconds = Column(Float)
    solver_type = Column(String(100), default="CBC")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary."""
        return {
            'id': self.id,
            'problem_id': self.problem_id,
            'solved_at': self.solved_at.isoformat() if self.solved_at else None,
            'status': self.status,
            'objective_value': self.objective_value,
            'variables': json.loads(self.variables) if self.variables else {},
            'shadow_prices': json.loads(self.shadow_prices) if self.shadow_prices else {},
            'slack_variables': json.loads(self.slack_variables) if self.slack_variables else {},
            'solver_time_seconds': self.solver_time_seconds,
            'solver_type': self.solver_type
        }

class SensitivityAnalysis(Base):
    """Table to store sensitivity analysis results."""
    __tablename__ = "sensitivity_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    problem_id = Column(Integer, nullable=False)
    solution_id = Column(Integer, nullable=False)
    analysis_type = Column(String(100))  # 'price', 'capacity', 'consumption'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Analysis parameters
    parameter_name = Column(String(255))
    parameter_index = Column(Integer)
    range_min = Column(Float)
    range_max = Column(Float)
    num_points = Column(Integer)
    
    # Results (stored as JSON)
    results_data = Column(Text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            'id': self.id,
            'problem_id': self.problem_id,
            'solution_id': self.solution_id,
            'analysis_type': self.analysis_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'parameter_name': self.parameter_name,
            'parameter_index': self.parameter_index,
            'range_min': self.range_min,
            'range_max': self.range_max,
            'num_points': self.num_points,
            'results_data': json.loads(self.results_data) if self.results_data else []
        }

class DatabaseManager:
    """Manager class for database operations."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            return True
        except Exception as e:
            print(f"Error creating tables: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def save_problem(self, name: str, description: str, products_data: List[Dict], 
                    resources_data: List[Dict], consumption_matrix: List[List[float]]) -> Optional[int]:
        """Save an optimization problem to the database."""
        try:
            with self.get_session() as session:
                problem = OptimizationProblem(
                    name=name,
                    description=description,
                    products_data=json.dumps(products_data),
                    resources_data=json.dumps(resources_data),
                    consumption_matrix=json.dumps(consumption_matrix),
                    num_products=len(products_data),
                    num_resources=len(resources_data)
                )
                session.add(problem)
                session.commit()
                session.refresh(problem)
                return problem.id
        except Exception as e:
            print(f"Error saving problem: {e}")
            return None
    
    def load_problem(self, problem_id: int) -> Optional[Dict[str, Any]]:
        """Load an optimization problem from the database."""
        try:
            with self.get_session() as session:
                problem = session.query(OptimizationProblem).filter(
                    OptimizationProblem.id == problem_id
                ).first()
                return problem.to_dict() if problem else None
        except Exception as e:
            print(f"Error loading problem: {e}")
            return None
    
    def list_problems(self) -> List[Dict[str, Any]]:
        """List all optimization problems."""
        try:
            with self.get_session() as session:
                problems = session.query(OptimizationProblem).order_by(
                    OptimizationProblem.updated_at.desc()
                ).all()
                return [problem.to_dict() for problem in problems]
        except Exception as e:
            print(f"Error listing problems: {e}")
            return []
    
    def delete_problem(self, problem_id: int) -> bool:
        """Delete a problem and all related data."""
        try:
            with self.get_session() as session:
                # Delete related solutions and analyses
                session.query(SensitivityAnalysis).filter(
                    SensitivityAnalysis.problem_id == problem_id
                ).delete()
                session.query(OptimizationSolution).filter(
                    OptimizationSolution.problem_id == problem_id
                ).delete()
                session.query(OptimizationProblem).filter(
                    OptimizationProblem.id == problem_id
                ).delete()
                session.commit()
                return True
        except Exception as e:
            print(f"Error deleting problem: {e}")
            return False
    
    def save_solution(self, problem_id: int, solution_data: Dict[str, Any], 
                     solver_time: float = 0.0) -> Optional[int]:
        """Save an optimization solution to the database."""
        try:
            with self.get_session() as session:
                solution = OptimizationSolution(
                    problem_id=problem_id,
                    status=solution_data.get('status', ''),
                    objective_value=solution_data.get('objective_value', 0.0),
                    variables=json.dumps(solution_data.get('variables', {})),
                    shadow_prices=json.dumps(solution_data.get('shadow_prices', {})),
                    slack_variables=json.dumps(solution_data.get('slack_variables', {})),
                    solver_time_seconds=solver_time
                )
                session.add(solution)
                session.commit()
                session.refresh(solution)
                return solution.id
        except Exception as e:
            print(f"Error saving solution: {e}")
            return None
    
    def load_solution(self, solution_id: int) -> Optional[Dict[str, Any]]:
        """Load a solution from the database."""
        try:
            with self.get_session() as session:
                solution = session.query(OptimizationSolution).filter(
                    OptimizationSolution.id == solution_id
                ).first()
                return solution.to_dict() if solution else None
        except Exception as e:
            print(f"Error loading solution: {e}")
            return None
    
    def list_solutions(self, problem_id: int) -> List[Dict[str, Any]]:
        """List all solutions for a problem."""
        try:
            with self.get_session() as session:
                solutions = session.query(OptimizationSolution).filter(
                    OptimizationSolution.problem_id == problem_id
                ).order_by(OptimizationSolution.solved_at.desc()).all()
                return [solution.to_dict() for solution in solutions]
        except Exception as e:
            print(f"Error listing solutions: {e}")
            return []
    
    def save_sensitivity_analysis(self, problem_id: int, solution_id: int, 
                                 analysis_type: str, parameter_name: str,
                                 parameter_index: int, range_min: float, 
                                 range_max: float, num_points: int,
                                 results_data: List[Dict]) -> Optional[int]:
        """Save sensitivity analysis results."""
        try:
            with self.get_session() as session:
                analysis = SensitivityAnalysis(
                    problem_id=problem_id,
                    solution_id=solution_id,
                    analysis_type=analysis_type,
                    parameter_name=parameter_name,
                    parameter_index=parameter_index,
                    range_min=range_min,
                    range_max=range_max,
                    num_points=num_points,
                    results_data=json.dumps(results_data)
                )
                session.add(analysis)
                session.commit()
                session.refresh(analysis)
                return analysis.id
        except Exception as e:
            print(f"Error saving sensitivity analysis: {e}")
            return None
    
    def list_sensitivity_analyses(self, problem_id: int) -> List[Dict[str, Any]]:
        """List all sensitivity analyses for a problem."""
        try:
            with self.get_session() as session:
                analyses = session.query(SensitivityAnalysis).filter(
                    SensitivityAnalysis.problem_id == problem_id
                ).order_by(SensitivityAnalysis.created_at.desc()).all()
                return [analysis.to_dict() for analysis in analyses]
        except Exception as e:
            print(f"Error listing sensitivity analyses: {e}")
            return []
    
    def get_problem_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.get_session() as session:
                num_problems = session.query(OptimizationProblem).count()
                num_solutions = session.query(OptimizationSolution).count()
                num_analyses = session.query(SensitivityAnalysis).count()
                
                # Get recent activity
                recent_problems = session.query(OptimizationProblem).order_by(
                    OptimizationProblem.updated_at.desc()
                ).limit(5).all()
                
                return {
                    'total_problems': num_problems,
                    'total_solutions': num_solutions,
                    'total_analyses': num_analyses,
                    'recent_problems': [p.to_dict() for p in recent_problems]
                }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_problems': 0,
                'total_solutions': 0,
                'total_analyses': 0,
                'recent_problems': []
            }

# Initialize database manager
db_manager = DatabaseManager()

# Initialize database tables
def init_database():
    """Initialize the database with all tables."""
    return db_manager.create_tables()

# Convenience functions for use in the main app
def save_problem_to_db(name: str, description: str, products_data: List[Dict], 
                      resources_data: List[Dict], consumption_matrix: List[List[float]]) -> Optional[int]:
    """Save problem to database - convenience function for main app."""
    return db_manager.save_problem(name, description, products_data, resources_data, consumption_matrix)

def load_problem_from_db(problem_id: int) -> Optional[Dict[str, Any]]:
    """Load problem from database - convenience function for main app."""
    return db_manager.load_problem(problem_id)

def list_problems_from_db() -> List[Dict[str, Any]]:
    """List all problems - convenience function for main app."""
    return db_manager.list_problems()

def save_solution_to_db(problem_id: int, solution_data: Dict[str, Any], solver_time: float = 0.0) -> Optional[int]:
    """Save solution to database - convenience function for main app."""
    return db_manager.save_solution(problem_id, solution_data, solver_time)