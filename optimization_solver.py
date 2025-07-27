import pulp
import numpy as np
from typing import Dict, List, Any

class OptimizationSolver:
    """
    A class to handle linear programming optimization problems using PuLP.
    Specifically designed for production planning problems.
    """
    
    def __init__(self, products_data: List[Dict], resources_data: List[Dict], consumption_matrix: List[List[float]]):
        """
        Initialize the optimization solver with problem data.
        
        Args:
            products_data: List of dictionaries containing product information
            resources_data: List of dictionaries containing resource information
            consumption_matrix: 2D list representing resource consumption per product
        """
        self.products_data = products_data
        self.resources_data = resources_data
        self.consumption_matrix = consumption_matrix
        self.num_products = len(products_data)
        self.num_resources = len(resources_data)
        
        # Initialize the optimization problem
        self.prob = None
        self.variables = {}
        
    def create_problem(self):
        """Create the linear programming problem formulation."""
        # Create the problem instance
        self.prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)
        
        # Create decision variables
        self.variables = {}
        for i in range(self.num_products):
            var_name = f"x_{i+1}"
            self.variables[var_name] = pulp.LpVariable(
                var_name, 
                lowBound=0, 
                cat='Continuous'
            )
        
        # Objective function: Maximize total profit
        profit_terms = []
        for i in range(self.num_products):
            var_name = f"x_{i+1}"
            profit = self.products_data[i]['profit']
            profit_terms.append(profit * self.variables[var_name])
        
        self.prob += pulp.lpSum(profit_terms), "Total_Profit"
        
        # Resource constraints
        for i in range(self.num_resources):
            resource_usage = []
            for j in range(self.num_products):
                var_name = f"x_{j+1}"
                consumption = self.consumption_matrix[i][j]
                resource_usage.append(consumption * self.variables[var_name])
            
            constraint_name = f"Resource_{i+1}_Constraint"
            capacity = self.resources_data[i]['capacity']
            self.prob += pulp.lpSum(resource_usage) <= capacity, constraint_name
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve the optimization problem.
        
        Returns:
            Dictionary containing solution information
        """
        # Create the problem if not already created
        if self.prob is None:
            self.create_problem()
        
        # Solve the problem
        try:
            if self.prob is not None:
                self.prob.solve(pulp.PULP_CBC_CMD(msg=False))
                
                # Extract solution
                status = pulp.LpStatus[self.prob.status]
            else:
                return {
                    'status': 'Error',
                    'error': 'Problem not properly initialized',
                    'objective_value': 0,
                    'variables': {},
                    'shadow_prices': {},
                    'slack_variables': {}
                }
            
            solution = {
                'status': status,
                'objective_value': 0,
                'variables': {},
                'shadow_prices': {},
                'slack_variables': {}
            }
            
            if status == 'Optimal' and self.prob is not None:
                # Get objective value
                solution['objective_value'] = pulp.value(self.prob.objective)
                
                # Get variable values
                for var_name, var in self.variables.items():
                    solution['variables'][var_name] = pulp.value(var)
                
                # Get shadow prices and slack variables
                for constraint in self.prob.constraints:
                    constraint_obj = self.prob.constraints[constraint]
                    if hasattr(constraint_obj, 'pi') and constraint_obj.pi is not None:
                        solution['shadow_prices'][constraint] = constraint_obj.pi
                    if hasattr(constraint_obj, 'slack') and constraint_obj.slack is not None:
                        solution['slack_variables'][constraint] = constraint_obj.slack
            
            return solution
            
        except Exception as e:
            return {
                'status': 'Error',
                'error': str(e),
                'objective_value': 0,
                'variables': {},
                'shadow_prices': {},
                'slack_variables': {}
            }
    
    def get_problem_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the problem structure.
        
        Returns:
            Dictionary containing problem summary information
        """
        if self.prob is None:
            self.create_problem()
        
        if self.prob is not None:
            return {
                'num_variables': len(self.variables),
                'num_constraints': len(self.prob.constraints),
                'variable_names': list(self.variables.keys()),
                'constraint_names': list(self.prob.constraints.keys()),
                'objective_sense': 'Maximize' if self.prob.sense == pulp.LpMaximize else 'Minimize'
            }
        else:
            return {
                'num_variables': 0,
                'num_constraints': 0,
                'variable_names': [],
                'constraint_names': [],
                'objective_sense': 'Unknown'
            }
    
    def validate_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the solution against constraints.
        
        Args:
            solution: Solution dictionary from solve()
            
        Returns:
            Dictionary containing validation results
        """
        if solution['status'] != 'Optimal':
            return {'valid': False, 'reason': f"Solution status is {solution['status']}"}
        
        validation_results = {
            'valid': True,
            'constraint_violations': [],
            'resource_usage': []
        }
        
        # Check resource constraints
        for i in range(self.num_resources):
            total_usage = 0
            for j in range(self.num_products):
                var_name = f"x_{j+1}"
                quantity = solution['variables'].get(var_name, 0)
                consumption = self.consumption_matrix[i][j]
                total_usage += quantity * consumption
            
            capacity = self.resources_data[i]['capacity']
            resource_name = self.resources_data[i]['name']
            
            usage_info = {
                'resource': resource_name,
                'usage': total_usage,
                'capacity': capacity,
                'utilization_pct': (total_usage / capacity) * 100 if capacity > 0 else 0,
                'slack': capacity - total_usage
            }
            validation_results['resource_usage'].append(usage_info)
            
            # Check for constraint violations
            if total_usage > capacity + 1e-6:  # Small tolerance for numerical errors
                validation_results['valid'] = False
                validation_results['constraint_violations'].append({
                    'resource': resource_name,
                    'usage': total_usage,
                    'capacity': capacity,
                    'violation': total_usage - capacity
                })
        
        # Check non-negativity constraints
        for var_name, value in solution['variables'].items():
            if value < -1e-6:  # Small tolerance for numerical errors
                validation_results['valid'] = False
                validation_results['constraint_violations'].append({
                    'variable': var_name,
                    'value': value,
                    'constraint': 'Non-negativity',
                    'violation': -value
                })
        
        return validation_results
    
    def get_model_string(self) -> str:
        """
        Get the string representation of the optimization model.
        
        Returns:
            String representation of the model
        """
        if self.prob is None:
            self.create_problem()
        
        return str(self.prob) if self.prob is not None else "Problem not initialized"
