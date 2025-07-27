import numpy as np
from typing import Dict, List, Any, Tuple
import copy

class SensitivityAnalyzer:
    """
    Class for performing sensitivity analysis on optimization problems.
    """
    
    def __init__(self, base_solver, products_data: List[Dict], 
                 resources_data: List[Dict], consumption_matrix: List[List[float]]):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            base_solver: The optimization solver instance
            products_data: List of product information
            resources_data: List of resource information
            consumption_matrix: Resource consumption matrix
        """
        self.base_solver = base_solver
        self.products_data = products_data
        self.resources_data = resources_data
        self.consumption_matrix = consumption_matrix
    
    def price_sensitivity(self, product_index: int, price_range: Tuple[int, int], 
                         num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Perform price sensitivity analysis for a specific product.
        
        Args:
            product_index: Index of the product to analyze
            price_range: Tuple of (min_percent, max_percent) for price variation
            num_points: Number of analysis points
            
        Returns:
            List of dictionaries containing sensitivity results
        """
        from optimization_solver import OptimizationSolver
        
        results = []
        base_price = self.products_data[product_index]['profit']
        
        # Create price variation range
        min_multiplier = 1 + (price_range[0] / 100)
        max_multiplier = 1 + (price_range[1] / 100)
        multipliers = np.linspace(min_multiplier, max_multiplier, num_points)
        
        for multiplier in multipliers:
            # Create modified product data
            modified_products = copy.deepcopy(self.products_data)
            modified_products[product_index]['profit'] = base_price * multiplier
            
            # Create new solver with modified data
            temp_solver = OptimizationSolver(
                modified_products, 
                self.resources_data, 
                self.consumption_matrix
            )
            
            # Solve the modified problem
            solution = temp_solver.solve()
            
            # Extract results
            result = {
                'price': base_price * multiplier,
                'price_multiplier': multiplier,
                'status': solution['status'],
                'profit': solution['objective_value'] if solution['status'] == 'Optimal' else 0,
                'quantity': 0
            }
            
            if solution['status'] == 'Optimal':
                var_name = f"x_{product_index + 1}"
                result['quantity'] = solution['variables'].get(var_name, 0)
                
                # Store all variable values for completeness
                result['all_quantities'] = solution['variables']
            
            results.append(result)
        
        return results
    
    def capacity_sensitivity(self, resource_index: int, capacity_range: Tuple[int, int], 
                           num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Perform capacity sensitivity analysis for a specific resource.
        
        Args:
            resource_index: Index of the resource to analyze
            capacity_range: Tuple of (min_percent, max_percent) for capacity variation
            num_points: Number of analysis points
            
        Returns:
            List of dictionaries containing sensitivity results
        """
        from optimization_solver import OptimizationSolver
        
        results = []
        base_capacity = self.resources_data[resource_index]['capacity']
        
        # Create capacity variation range
        min_multiplier = 1 + (capacity_range[0] / 100)
        max_multiplier = 1 + (capacity_range[1] / 100)
        multipliers = np.linspace(min_multiplier, max_multiplier, num_points)
        
        for multiplier in multipliers:
            # Create modified resource data
            modified_resources = copy.deepcopy(self.resources_data)
            modified_resources[resource_index]['capacity'] = base_capacity * multiplier
            
            # Create new solver with modified data
            temp_solver = OptimizationSolver(
                self.products_data, 
                modified_resources, 
                self.consumption_matrix
            )
            
            # Solve the modified problem
            solution = temp_solver.solve()
            
            # Extract results
            result = {
                'capacity': base_capacity * multiplier,
                'capacity_multiplier': multiplier,
                'status': solution['status'],
                'profit': solution['objective_value'] if solution['status'] == 'Optimal' else 0
            }
            
            if solution['status'] == 'Optimal':
                result['quantities'] = solution['variables']
                
                # Calculate resource utilization
                used = 0
                for j, product in enumerate(self.products_data):
                    var_name = f"x_{j+1}"
                    quantity = solution['variables'].get(var_name, 0)
                    used += quantity * self.consumption_matrix[resource_index][j]
                
                result['resource_used'] = used
                result['utilization_pct'] = (used / (base_capacity * multiplier)) * 100 if (base_capacity * multiplier) > 0 else 0
            
            results.append(result)
        
        return results
    
    def consumption_sensitivity(self, product_index: int, resource_index: int, 
                              consumption_range: Tuple[int, int], num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Perform sensitivity analysis on resource consumption coefficients.
        
        Args:
            product_index: Index of the product
            resource_index: Index of the resource
            consumption_range: Tuple of (min_percent, max_percent) for consumption variation
            num_points: Number of analysis points
            
        Returns:
            List of dictionaries containing sensitivity results
        """
        from optimization_solver import OptimizationSolver
        
        results = []
        base_consumption = self.consumption_matrix[resource_index][product_index]
        
        # Create consumption variation range
        min_multiplier = 1 + (consumption_range[0] / 100)
        max_multiplier = 1 + (consumption_range[1] / 100)
        multipliers = np.linspace(min_multiplier, max_multiplier, num_points)
        
        for multiplier in multipliers:
            # Create modified consumption matrix
            modified_matrix = copy.deepcopy(self.consumption_matrix)
            modified_matrix[resource_index][product_index] = base_consumption * multiplier
            
            # Create new solver with modified data
            temp_solver = OptimizationSolver(
                self.products_data, 
                self.resources_data, 
                modified_matrix
            )
            
            # Solve the modified problem
            solution = temp_solver.solve()
            
            # Extract results
            result = {
                'consumption': base_consumption * multiplier,
                'consumption_multiplier': multiplier,
                'status': solution['status'],
                'profit': solution['objective_value'] if solution['status'] == 'Optimal' else 0
            }
            
            if solution['status'] == 'Optimal':
                result['quantities'] = solution['variables']
                var_name = f"x_{product_index + 1}"
                result['product_quantity'] = solution['variables'].get(var_name, 0)
            
            results.append(result)
        
        return results
    
    def multi_parameter_sensitivity(self, parameters: List[Dict[str, Any]], 
                                  num_points: int = 10) -> List[Dict[str, Any]]:
        """
        Perform multi-parameter sensitivity analysis.
        
        Args:
            parameters: List of parameter specifications
            num_points: Number of analysis points
            
        Returns:
            List of dictionaries containing sensitivity results
        """
        from optimization_solver import OptimizationSolver
        
        results = []
        
        # Generate parameter combinations (simplified version - equal steps for all parameters)
        multiplier_range = np.linspace(0.5, 1.5, num_points)
        
        for multiplier in multiplier_range:
            # Apply all parameter changes
            modified_products = copy.deepcopy(self.products_data)
            modified_resources = copy.deepcopy(self.resources_data)
            modified_matrix = copy.deepcopy(self.consumption_matrix)
            
            parameter_changes = {}
            
            for param in parameters:
                param_type = param['type']
                param_index = param['index']
                
                if param_type == 'price':
                    base_value = self.products_data[param_index]['profit']
                    modified_products[param_index]['profit'] = base_value * multiplier
                    parameter_changes[f'price_{param_index}'] = base_value * multiplier
                
                elif param_type == 'capacity':
                    base_value = self.resources_data[param_index]['capacity']
                    modified_resources[param_index]['capacity'] = base_value * multiplier
                    parameter_changes[f'capacity_{param_index}'] = base_value * multiplier
                
                elif param_type == 'consumption':
                    resource_idx = param['resource_index']
                    product_idx = param['product_index']
                    base_value = self.consumption_matrix[resource_idx][product_idx]
                    modified_matrix[resource_idx][product_idx] = base_value * multiplier
                    parameter_changes[f'consumption_{resource_idx}_{product_idx}'] = base_value * multiplier
            
            # Create new solver with all modifications
            temp_solver = OptimizationSolver(modified_products, modified_resources, modified_matrix)
            
            # Solve the modified problem
            solution = temp_solver.solve()
            
            # Extract results
            result = {
                'multiplier': multiplier,
                'parameter_changes': parameter_changes,
                'status': solution['status'],
                'profit': solution['objective_value'] if solution['status'] == 'Optimal' else 0
            }
            
            if solution['status'] == 'Optimal':
                result['quantities'] = solution['variables']
            
            results.append(result)
        
        return results
    
    def shadow_price_analysis(self) -> Dict[str, Any]:
        """
        Analyze shadow prices and their economic interpretation.
        
        Returns:
            Dictionary containing shadow price analysis
        """
        # Solve the base problem to get shadow prices
        base_solution = self.base_solver.solve()
        
        if base_solution['status'] != 'Optimal':
            return {'status': 'Error', 'message': 'Base problem not optimal'}
        
        analysis = {
            'status': 'Success',
            'shadow_prices': base_solution.get('shadow_prices', {}),
            'interpretations': [],
            'recommendations': []
        }
        
        # Interpret shadow prices
        for constraint_name, shadow_price in base_solution.get('shadow_prices', {}).items():
            if 'Resource' in constraint_name and shadow_price is not None:
                resource_index = int(constraint_name.split('_')[1]) - 1
                resource_name = self.resources_data[resource_index]['name']
                
                interpretation = {
                    'resource': resource_name,
                    'shadow_price': shadow_price,
                    'interpretation': ''
                }
                
                if shadow_price > 1e-6:  # Positive shadow price
                    interpretation['interpretation'] = f"Increasing {resource_name} capacity by 1 unit would increase profit by ${shadow_price:.2f}"
                    analysis['recommendations'].append(f"Consider increasing {resource_name} capacity - high value constraint")
                elif shadow_price < -1e-6:  # Negative shadow price (shouldn't happen in this context)
                    interpretation['interpretation'] = f"Unusual negative shadow price for {resource_name}"
                else:  # Zero shadow price
                    interpretation['interpretation'] = f"{resource_name} is not a binding constraint - has slack capacity"
                    analysis['recommendations'].append(f"{resource_name} has excess capacity - could be reduced or reallocated")
                
                analysis['interpretations'].append(interpretation)
        
        return analysis
    
    def bottleneck_analysis(self) -> Dict[str, Any]:
        """
        Identify and analyze bottleneck resources.
        
        Returns:
            Dictionary containing bottleneck analysis
        """
        from optimization_solver import OptimizationSolver
        
        # Solve base problem
        base_solution = self.base_solver.solve()
        
        if base_solution['status'] != 'Optimal':
            return {'status': 'Error', 'message': 'Base problem not optimal'}
        
        # Calculate resource utilization
        resource_analysis = []
        
        for i, resource in enumerate(self.resources_data):
            used = 0
            for j, product in enumerate(self.products_data):
                var_name = f"x_{j+1}"
                quantity = base_solution['variables'].get(var_name, 0)
                used += quantity * self.consumption_matrix[i][j]
            
            utilization_pct = (used / resource['capacity']) * 100 if resource['capacity'] > 0 else 0
            slack = resource['capacity'] - used
            
            resource_info = {
                'resource': resource['name'],
                'capacity': resource['capacity'],
                'used': used,
                'slack': slack,
                'utilization_pct': utilization_pct,
                'is_bottleneck': utilization_pct > 95,  # Consider >95% as bottleneck
                'is_binding': abs(slack) < 1e-6
            }
            
            resource_analysis.append(resource_info)
        
        # Sort by utilization
        resource_analysis.sort(key=lambda x: x['utilization_pct'], reverse=True)
        
        # Identify bottlenecks
        bottlenecks = [r for r in resource_analysis if r['is_bottleneck']]
        underutilized = [r for r in resource_analysis if r['utilization_pct'] < 50]
        
        analysis = {
            'status': 'Success',
            'resource_analysis': resource_analysis,
            'bottlenecks': bottlenecks,
            'underutilized': underutilized,
            'recommendations': []
        }
        
        # Generate recommendations
        if bottlenecks:
            analysis['recommendations'].append("Consider increasing capacity for bottleneck resources")
            for bottleneck in bottlenecks:
                analysis['recommendations'].append(f"- {bottleneck['resource']}: {bottleneck['utilization_pct']:.1f}% utilized")
        
        if underutilized:
            analysis['recommendations'].append("Consider reallocating or reducing underutilized resources")
            for under in underutilized:
                analysis['recommendations'].append(f"- {under['resource']}: {under['utilization_pct']:.1f}% utilized")
        
        return analysis
