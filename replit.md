# Business Optimization with Linear Programming

## Overview

This is a Streamlit-based web application for solving business optimization problems using linear programming. The application focuses on production planning scenarios where companies need to optimize their production mix to maximize profit while respecting resource constraints. It provides an interactive interface for problem setup, mathematical formulation, solution visualization, and sensitivity analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for rapid prototyping and interactive data applications
- **Layout**: Multi-page application with sidebar navigation
- **State Management**: Streamlit session state for maintaining solver instance and solution data across page interactions
- **Visualization**: Plotly for interactive charts and graphs

### Backend Architecture
- **Core Logic**: Object-oriented design with separate classes for different concerns
- **Optimization Engine**: PuLP library for linear programming solver functionality
- **Modular Design**: Separate modules for optimization solving, visualization, and sensitivity analysis

## Key Components

### 1. Main Application (app.py)
- **Purpose**: Entry point and UI orchestration
- **Responsibilities**: Page routing, session state management, user interface layout
- **Architecture Decision**: Single-file approach for UI with imports from specialized modules to maintain separation of concerns

### 2. Optimization Solver (optimization_solver.py)
- **Purpose**: Core linear programming problem formulation and solving
- **Technology**: PuLP library for mathematical optimization
- **Design Pattern**: Class-based approach for encapsulating problem state and methods
- **Key Features**: Decision variable creation, constraint formulation, objective function definition

### 3. Visualization Utils (visualization_utils.py)
- **Purpose**: Generate interactive charts and graphs for results
- **Technology**: Plotly for dynamic, interactive visualizations
- **Architecture**: Utility class pattern for reusable visualization components
- **Chart Types**: Production quantity bars, resource utilization, profit analysis

### 4. Sensitivity Analysis (sensitivity_analysis.py)
- **Purpose**: Perform what-if analysis on optimization parameters
- **Approach**: Parametric analysis by varying input values and re-solving
- **Design**: Separate analyzer class to maintain clean separation from core optimization logic

## Data Flow

1. **Input Collection**: User provides product data, resource constraints, and consumption matrix through Streamlit interface
2. **Problem Formulation**: OptimizationSolver creates PuLP problem instance with variables and constraints
3. **Solution**: PuLP solver engine computes optimal solution
4. **Visualization**: VisualizationUtils generates interactive charts from solution data
5. **Analysis**: SensitivityAnalyzer performs parameter variations for deeper insights
6. **Export**: Results can be exported for external use

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for rapid data app development
- **PuLP**: Linear programming library for optimization solving
- **Plotly**: Interactive visualization library for charts and graphs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing support

### Rationale for Technology Choices
- **Streamlit over Flask/Django**: Faster development for data-focused applications with built-in widgets
- **PuLP over SciPy**: More intuitive API for linear programming with multiple solver backends
- **Plotly over Matplotlib**: Interactive visualizations enhance user experience for business applications

## Deployment Strategy

### Current Architecture
- **Target Platform**: Streamlit Cloud or similar Python hosting platforms
- **Dependencies**: Managed through requirements.txt (implied but not present in repository)
- **Configuration**: Streamlit page configuration for responsive layout

### Scalability Considerations
- **State Management**: Session state combined with PostgreSQL persistence for robust data management
- **Performance**: In-memory solving with database caching for optimal performance
- **Database Storage**: PostgreSQL integration enables problem persistence, history tracking, and multi-user scenarios

## Development Notes

### Code Organization
- **Modular Design**: Clear separation between UI, optimization logic, visualization, and analysis
- **Object-Oriented**: Class-based architecture for maintainable and extensible code
- **Type Hints**: Partial implementation suggests intention for better code documentation

### Completed Components
- **Database Integration**: PostgreSQL database with complete data persistence
- **Problem Management**: Save, load, delete, and browse optimization problems
- **Solution History**: Track multiple solutions with timestamps and performance metrics
- **Dependencies**: All required packages properly installed and configured

### Remaining Enhancements
- **Error Handling**: Enhanced error handling for invalid inputs and solver failures
- **Data Validation**: Input validation for business constraints and mathematical feasibility
- **Testing**: Unit tests for optimization logic and visualization components

### Recent Updates (July 26, 2025)
- **Database Integration**: Added PostgreSQL database for persistent storage of optimization problems, solutions, and sensitivity analyses
- **Data Management**: New Database Management page with full CRUD operations for problems and solutions
- **Problem History**: Track multiple solutions and analyses for each optimization problem
- **Enhanced Workflow**: Users can now save, load, and compare different optimization scenarios

### Extension Points
- **Problem Types**: Architecture supports extending to other optimization problem types
- **Solvers**: PuLP backend allows switching between different optimization solvers
- **Export Formats**: Multiple export options can be added for different business needs
- **Database Schema**: Extensible database design supports additional problem types and analysis methods