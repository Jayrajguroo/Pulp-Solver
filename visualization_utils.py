import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class VisualizationUtils:
    """
    Utility class for creating visualizations of optimization results.
    """
    
    def __init__(self, solution: Dict[str, Any], products_data: List[Dict], 
                 resources_data: List[Dict], consumption_matrix: List[List[float]]):
        """
        Initialize the visualization utility.
        
        Args:
            solution: Solution dictionary from optimization solver
            products_data: List of product information
            resources_data: List of resource information
            consumption_matrix: Resource consumption matrix
        """
        self.solution = solution
        self.products_data = products_data
        self.resources_data = resources_data
        self.consumption_matrix = consumption_matrix
    
    def create_production_chart(self) -> go.Figure:
        """Create a bar chart showing optimal production quantities."""
        if self.solution['status'] != 'Optimal':
            return go.Figure().add_annotation(text="No optimal solution available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        products = [p['name'] for p in self.products_data]
        quantities = []
        
        for i, product in enumerate(self.products_data):
            var_name = f"x_{i+1}"
            quantity = self.solution['variables'].get(var_name, 0)
            quantities.append(quantity)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=products,
                y=quantities,
                text=[f"{q:.1f}" for q in quantities],
                textposition='auto',
                marker_color='lightblue',
                marker_line_color='darkblue',
                marker_line_width=2
            )
        ])
        
        fig.update_layout(
            title="Optimal Production Quantities",
            xaxis_title="Products",
            yaxis_title="Quantity to Produce",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_resource_utilization_chart(self) -> go.Figure:
        """Create a horizontal bar chart showing resource utilization."""
        if self.solution['status'] != 'Optimal':
            return go.Figure().add_annotation(text="No optimal solution available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        resources = [r['name'] for r in self.resources_data]
        capacities = [r['capacity'] for r in self.resources_data]
        used_amounts = []
        utilization_pcts = []
        
        for i, resource in enumerate(self.resources_data):
            used = 0
            for j, product in enumerate(self.products_data):
                var_name = f"x_{j+1}"
                quantity = self.solution['variables'].get(var_name, 0)
                used += quantity * self.consumption_matrix[i][j]
            
            used_amounts.append(used)
            utilization_pct = (used / resource['capacity']) * 100 if resource['capacity'] > 0 else 0
            utilization_pcts.append(utilization_pct)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add capacity bars (background)
        fig.add_trace(go.Bar(
            y=resources,
            x=capacities,
            orientation='h',
            name='Total Capacity',
            marker_color='lightgray',
            opacity=0.7
        ))
        
        # Add utilization bars
        colors = ['red' if pct > 95 else 'orange' if pct > 80 else 'green' for pct in utilization_pcts]
        fig.add_trace(go.Bar(
            y=resources,
            x=used_amounts,
            orientation='h',
            name='Used',
            marker_color=colors,
            text=[f"{pct:.1f}%" for pct in utilization_pcts],
            textposition='inside'
        ))
        
        fig.update_layout(
            title="Resource Utilization",
            xaxis_title="Capacity / Usage",
            yaxis_title="Resources",
            barmode='overlay',
            height=max(300, len(resources) * 50)
        )
        
        return fig
    
    def create_profit_breakdown_chart(self) -> go.Figure:
        """Create a pie chart showing profit contribution by product."""
        if self.solution['status'] != 'Optimal':
            return go.Figure().add_annotation(text="No optimal solution available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        products = []
        profits = []
        
        for i, product in enumerate(self.products_data):
            var_name = f"x_{i+1}"
            quantity = self.solution['variables'].get(var_name, 0)
            profit_contribution = quantity * product['profit']
            
            if profit_contribution > 0:  # Only include products with positive contribution
                products.append(product['name'])
                profits.append(profit_contribution)
        
        if not products:
            return go.Figure().add_annotation(text="No profit contributions to display", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=products,
                values=profits,
                textinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent}<br>$%{value:.0f}',
                hovertemplate='<b>%{label}</b><br>Profit: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Profit Contribution by Product",
            height=500
        )
        
        return fig
    
    def create_feasible_region_chart(self) -> go.Figure:
        """Create a 2D feasible region chart (only works for 2 products)."""
        if len(self.products_data) != 2:
            return go.Figure().add_annotation(
                text="Feasible region visualization only available for 2-product problems",
                xref="paper", yref="paper", x=0.5, y=0.5
            )
        
        if self.solution['status'] != 'Optimal':
            return go.Figure().add_annotation(text="No optimal solution available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Get the optimal solution point
        x1_opt = self.solution['variables'].get('x_1', 0)
        x2_opt = self.solution['variables'].get('x_2', 0)
        
        # Create a grid for plotting
        max_x1 = max(100, x1_opt * 1.5)
        max_x2 = max(100, x2_opt * 1.5)
        
        fig = go.Figure()
        
        # Plot constraint lines
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, resource in enumerate(self.resources_data):
            # Get constraint coefficients
            a1 = self.consumption_matrix[i][0]  # coefficient for x1
            a2 = self.consumption_matrix[i][1]  # coefficient for x2
            b = resource['capacity']  # right-hand side
            
            # Calculate constraint line points
            if a2 != 0:
                # Line equation: x2 = (b - a1*x1) / a2
                x1_line = np.linspace(0, max_x1, 100)
                x2_line = (b - a1 * x1_line) / a2
                
                # Only keep positive values
                valid_mask = (x1_line >= 0) & (x2_line >= 0)
                x1_line = x1_line[valid_mask]
                x2_line = x2_line[valid_mask]
                
                if len(x1_line) > 0:
                    fig.add_trace(go.Scatter(
                        x=x1_line,
                        y=x2_line,
                        mode='lines',
                        name=f'{resource["name"]} Constraint',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            elif a1 != 0:
                # Vertical line: x1 = b/a1
                x1_const = b / a1
                if x1_const >= 0:
                    fig.add_trace(go.Scatter(
                        x=[x1_const, x1_const],
                        y=[0, max_x2],
                        mode='lines',
                        name=f'{resource["name"]} Constraint',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
        
        # Add optimal solution point
        fig.add_trace(go.Scatter(
            x=[x1_opt],
            y=[x2_opt],
            mode='markers',
            name='Optimal Solution',
            marker=dict(color='black', size=12, symbol='star'),
            text=[f'Optimal: ({x1_opt:.2f}, {x2_opt:.2f})<br>Profit: ${self.solution["objective_value"]:.2f}'],
            textposition='top center'
        ))
        
        # Add objective function contours
        profit1 = self.products_data[0]['profit']
        profit2 = self.products_data[1]['profit']
        
        # Add a few iso-profit lines
        target_profits = [self.solution['objective_value'] * 0.5, 
                         self.solution['objective_value'] * 0.75,
                         self.solution['objective_value']]
        
        for j, target_profit in enumerate(target_profits):
            if profit2 != 0:
                x1_profit = np.linspace(0, max_x1, 100)
                x2_profit = (target_profit - profit1 * x1_profit) / profit2
                
                valid_mask = (x1_profit >= 0) & (x2_profit >= 0) & (x2_profit <= max_x2)
                x1_profit = x1_profit[valid_mask]
                x2_profit = x2_profit[valid_mask]
                
                if len(x1_profit) > 0:
                    opacity = 0.3 + j * 0.2
                    fig.add_trace(go.Scatter(
                        x=x1_profit,
                        y=x2_profit,
                        mode='lines',
                        name=f'Profit = ${target_profit:.0f}',
                        line=dict(color='gray', width=1, dash='dash'),
                        opacity=opacity,
                        showlegend=j == len(target_profits) - 1
                    ))
        
        fig.update_layout(
            title="Feasible Region and Optimal Solution",
            xaxis_title=f"x₁ ({self.products_data[0]['name']})",
            yaxis_title=f"x₂ ({self.products_data[1]['name']})",
            xaxis=dict(range=[0, max_x1]),
            yaxis=dict(range=[0, max_x2]),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_sensitivity_comparison_chart(self, base_solution: Dict, sensitivity_results: List[Dict], 
                                          parameter_name: str) -> go.Figure:
        """Create a chart comparing sensitivity analysis results."""
        if not sensitivity_results:
            return go.Figure().add_annotation(text="No sensitivity results available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Extract data for plotting
        parameters = [r['parameter_value'] for r in sensitivity_results]
        profits = [r['profit'] for r in sensitivity_results]
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=parameters,
            y=profits,
            mode='lines+markers',
            name='Total Profit',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add baseline point
        base_param = next((r['parameter_value'] for r in sensitivity_results 
                          if abs(r['profit'] - base_solution['objective_value']) < 1e-6), None)
        
        if base_param is not None:
            fig.add_trace(go.Scatter(
                x=[base_param],
                y=[base_solution['objective_value']],
                mode='markers',
                name='Base Case',
                marker=dict(color='red', size=12, symbol='star')
            ))
        
        fig.update_layout(
            title=f"Sensitivity Analysis - {parameter_name}",
            xaxis_title=parameter_name,
            yaxis_title="Total Profit ($)",
            hovermode='x unified',
            height=400
        )
        
        return fig
