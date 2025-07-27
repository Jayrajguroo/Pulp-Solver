import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from optimization_solver import OptimizationSolver
from visualization_utils import VisualizationUtils
from sensitivity_analysis import SensitivityAnalyzer
from database import (init_database, save_problem_to_db, load_problem_from_db, 
                     list_problems_from_db, save_solution_to_db, db_manager)

# Page configuration
st.set_page_config(
    page_title="Business Optimization with Linear Programming",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = init_database()

# Initialize session state
if 'solver' not in st.session_state:
    st.session_state.solver = None
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'problem_solved' not in st.session_state:
    st.session_state.problem_solved = False
if 'current_problem_id' not in st.session_state:
    st.session_state.current_problem_id = None
if 'current_solution_id' not in st.session_state:
    st.session_state.current_solution_id = None

def main():
    st.title("🏭 Business Optimization with Linear Programming")
    st.markdown("""
    This application demonstrates how to solve business optimization problems using linear programming.
    We'll work through a **Production Planning Problem** where a company needs to optimize their 
    production mix to maximize profit while respecting resource constraints.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Problem Setup", "Mathematical Formulation", "Solution & Results", "Visualization", "Sensitivity Analysis", "Database Management", "Export Results"]
    )
    
    if page == "Problem Setup":
        problem_setup_page()
    elif page == "Mathematical Formulation":
        mathematical_formulation_page()
    elif page == "Solution & Results":
        solution_results_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Sensitivity Analysis":
        sensitivity_analysis_page()
    elif page == "Database Management":
        database_management_page()
    elif page == "Export Results":
        export_results_page()

def problem_setup_page():
    st.header("📋 Problem Setup")
    
    st.markdown("""
    ### Production Planning Scenario
    
    A manufacturing company produces multiple products and needs to determine the optimal production 
    quantities to maximize profit while respecting resource constraints such as:
    - Raw material availability
    - Labor hours
    - Machine capacity
    - Storage space
    """)
    
    # Problem configuration
    st.subheader("Configure Your Problem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Products")
        num_products = st.slider("Number of products", 2, 5, 3)
        
        products_data = []
        for i in range(num_products):
            st.markdown(f"**Product {i+1}:**")
            name = st.text_input(f"Product name", f"Product_{chr(65+i)}", key=f"product_name_{i}")
            profit = st.number_input(f"Profit per unit ($)", 0.0, 1000.0, 50.0 + i*25, key=f"profit_{i}")
            products_data.append({"name": name, "profit": profit})
    
    with col2:
        st.markdown("#### Resources")
        num_resources = st.slider("Number of resources", 2, 4, 3)
        
        resources_data = []
        for i in range(num_resources):
            st.markdown(f"**Resource {i+1}:**")
            name = st.text_input(f"Resource name", f"Resource_{i+1}", key=f"resource_name_{i}")
            capacity = st.number_input(f"Available capacity", 0, 10000, 1000 + i*500, key=f"capacity_{i}")
            resources_data.append({"name": name, "capacity": capacity})
    
    # Resource consumption matrix
    st.subheader("Resource Consumption Matrix")
    st.markdown("Define how much of each resource is required to produce one unit of each product:")
    
    consumption_matrix = []
    cols = st.columns(num_products + 1)
    
    # Header row
    cols[0].markdown("**Resources**")
    for j, product in enumerate(products_data):
        cols[j+1].markdown(f"**{product['name']}**")
    
    # Data rows
    for i, resource in enumerate(resources_data):
        row = []
        cols[0].markdown(f"{resource['name']}")
        for j in range(num_products):
            consumption = cols[j+1].number_input(
                f"Usage",
                0.0, 100.0, 1.0 + (i+j)*0.5,
                key=f"consumption_{i}_{j}",
                label_visibility="collapsed"
            )
            row.append(consumption)
        consumption_matrix.append(row)
    
    # Store data in session state and save to database
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Configure Problem", type="primary"):
            st.session_state.products_data = products_data
            st.session_state.resources_data = resources_data
            st.session_state.consumption_matrix = consumption_matrix
            st.session_state.solver = OptimizationSolver(products_data, resources_data, consumption_matrix)
            st.success("Problem configured successfully! Navigate to 'Mathematical Formulation' to see the problem structure.")
            st.rerun()
    
    with col2:
        if st.button("Save Problem to Database"):
            if hasattr(st.session_state, 'products_data'):
                problem_name = st.text_input("Problem Name", f"Problem_{len(list_problems_from_db()) + 1}")
                problem_desc = st.text_area("Description", "Production planning optimization problem")
                
                if problem_name:
                    problem_id = save_problem_to_db(
                        problem_name, 
                        problem_desc,
                        st.session_state.products_data,
                        st.session_state.resources_data,
                        st.session_state.consumption_matrix
                    )
                    if problem_id:
                        st.session_state.current_problem_id = problem_id
                        st.success(f"Problem saved to database with ID: {problem_id}")
                    else:
                        st.error("Failed to save problem to database")
            else:
                st.warning("Please configure a problem first")
    
    # Display current configuration
    if hasattr(st.session_state, 'products_data'):
        st.subheader("Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Products:**")
            products_df = pd.DataFrame(st.session_state.products_data)
            st.dataframe(products_df, use_container_width=True)
        
        with col2:
            st.markdown("**Resources:**")
            resources_df = pd.DataFrame(st.session_state.resources_data)
            st.dataframe(resources_df, use_container_width=True)
        
        st.markdown("**Resource Consumption Matrix:**")
        consumption_df = pd.DataFrame(
            st.session_state.consumption_matrix
        )
        consumption_df.columns = [p['name'] for p in st.session_state.products_data]
        consumption_df.index = [r['name'] for r in st.session_state.resources_data]
        st.dataframe(consumption_df, use_container_width=True)

def mathematical_formulation_page():
    st.header("📐 Mathematical Formulation")
    
    if not hasattr(st.session_state, 'solver'):
        st.warning("Please configure the problem in the 'Problem Setup' section first.")
        return
    
    solver = st.session_state.solver
    
    st.markdown("""
    ### Linear Programming Model
    
    Our production planning problem can be formulated as a linear programming model:
    """)
    
    # Objective function
    st.subheader("Objective Function")
    st.markdown("**Maximize Total Profit:**")
    
    objective_parts = []
    for i, product in enumerate(st.session_state.products_data):
        objective_parts.append(f"{product['profit']}x_{i+1}")
    
    objective_formula = "Maximize: " + " + ".join(objective_parts)
    st.latex(objective_formula.replace("Maximize: ", "\\text{Maximize: } "))
    
    st.markdown("Where:")
    for i, product in enumerate(st.session_state.products_data):
        st.markdown(f"- x_{i+1} = units of {product['name']} to produce")
    
    # Constraints
    st.subheader("Constraints")
    
    st.markdown("**Resource Constraints:**")
    for i, resource in enumerate(st.session_state.resources_data):
        constraint_parts = []
        for j in range(len(st.session_state.products_data)):
            coeff = st.session_state.consumption_matrix[i][j]
            constraint_parts.append(f"{coeff}x_{j+1}")
        
        constraint_formula = " + ".join(constraint_parts) + f" \\leq {resource['capacity']}"
        st.latex(constraint_formula + f"\\quad \\text{{({resource['name']})}} ")
    
    st.markdown("**Non-negativity Constraints:**")
    non_neg_constraints = []
    for i in range(len(st.session_state.products_data)):
        non_neg_constraints.append(f"x_{i+1} \\geq 0")
    
    st.latex(", \\quad ".join(non_neg_constraints))
    
    # Problem summary
    st.subheader("Problem Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Decision Variables", len(st.session_state.products_data))
    
    with col2:
        st.metric("Constraints", len(st.session_state.resources_data))
    
    with col3:
        st.metric("Problem Type", "Linear Programming")
    
    # Solution button
    if st.button("Solve Optimization Problem", type="primary"):
        with st.spinner("Solving..."):
            start_time = time.time()
            solution = solver.solve()
            solve_time = time.time() - start_time
            
            st.session_state.solution = solution
            st.session_state.problem_solved = True
            
            # Save solution to database if we have a current problem
            if st.session_state.current_problem_id and solution['status'] == 'Optimal':
                solution_id = save_solution_to_db(
                    st.session_state.current_problem_id,
                    solution,
                    solve_time
                )
                if solution_id:
                    st.session_state.current_solution_id = solution_id
            
        if solution['status'] == 'Optimal':
            st.success(f"✅ Optimal solution found! Maximum profit: ${solution['objective_value']:,.2f}")
            if st.session_state.current_solution_id:
                st.info(f"Solution saved to database with ID: {st.session_state.current_solution_id}")
        else:
            st.error(f"❌ Problem status: {solution['status']}")
        
        st.rerun()

def solution_results_page():
    st.header("🎯 Solution & Results")
    
    if not st.session_state.problem_solved:
        st.warning("Please solve the problem in the 'Mathematical Formulation' section first.")
        return
    
    solution = st.session_state.solution
    
    if solution['status'] != 'Optimal':
        st.error(f"Problem could not be solved optimally. Status: {solution['status']}")
        return
    
    # Solution summary
    st.subheader("Optimal Solution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Maximum Profit", f"${solution['objective_value']:,.2f}")
        st.metric("Problem Status", solution['status'])
    
    with col2:
        total_production = sum(solution['variables'].values())
        st.metric("Total Units Produced", f"{total_production:,.0f}")
    
    # Production plan
    st.subheader("Optimal Production Plan")
    
    production_data = []
    for i, product in enumerate(st.session_state.products_data):
        var_name = f"x_{i+1}"
        quantity = solution['variables'].get(var_name, 0)
        revenue = quantity * product['profit']
        production_data.append({
            'Product': product['name'],
            'Optimal Quantity': quantity,
            'Unit Profit': f"${product['profit']:.2f}",
            'Total Revenue': f"${revenue:.2f}"
        })
    
    production_df = pd.DataFrame(production_data)
    st.dataframe(production_df, use_container_width=True)
    
    # Resource utilization
    st.subheader("Resource Utilization")
    
    resource_usage = []
    for i, resource in enumerate(st.session_state.resources_data):
        used = 0
        for j, product in enumerate(st.session_state.products_data):
            var_name = f"x_{j+1}"
            quantity = solution['variables'].get(var_name, 0)
            used += quantity * st.session_state.consumption_matrix[i][j]
        
        utilization_pct = (used / resource['capacity']) * 100 if resource['capacity'] > 0 else 0
        slack = resource['capacity'] - used
        
        resource_usage.append({
            'Resource': resource['name'],
            'Capacity': resource['capacity'],
            'Used': f"{used:.2f}",
            'Slack': f"{slack:.2f}",
            'Utilization %': f"{utilization_pct:.1f}%"
        })
    
    resource_df = pd.DataFrame(resource_usage)
    st.dataframe(resource_df, use_container_width=True)
    
    # Insights
    st.subheader("Key Insights")
    
    insights = []
    
    # Find bottleneck resources
    max_utilization = 0
    bottleneck_resource = ""
    for i, resource in enumerate(st.session_state.resources_data):
        used = sum(solution['variables'].get(f"x_{j+1}", 0) * st.session_state.consumption_matrix[i][j] 
                  for j in range(len(st.session_state.products_data)))
        utilization = (used / resource['capacity']) * 100 if resource['capacity'] > 0 else 0
        if utilization > max_utilization:
            max_utilization = utilization
            bottleneck_resource = resource['name']
    
    insights.append(f"🔍 **Bottleneck Resource**: {bottleneck_resource} ({max_utilization:.1f}% utilization)")
    
    # Find most profitable product in solution
    max_production = 0
    top_product = ""
    for i, product in enumerate(st.session_state.products_data):
        quantity = solution['variables'].get(f"x_{i+1}", 0)
        if quantity > max_production:
            max_production = quantity
            top_product = product['name']
    
    if max_production > 0:
        insights.append(f"📈 **Top Production**: {top_product} ({max_production:.0f} units)")
    
    # Identify unused resources
    unused_resources = []
    for i, resource in enumerate(st.session_state.resources_data):
        used = sum(solution['variables'].get(f"x_{j+1}", 0) * st.session_state.consumption_matrix[i][j] 
                  for j in range(len(st.session_state.products_data)))
        if used < resource['capacity'] * 0.95:  # Less than 95% utilized
            unused_resources.append(resource['name'])
    
    if unused_resources:
        insights.append(f"💡 **Underutilized Resources**: {', '.join(unused_resources)}")
    
    for insight in insights:
        st.markdown(insight)

def visualization_page():
    st.header("📊 Visualization")
    
    if not st.session_state.problem_solved:
        st.warning("Please solve the problem first.")
        return
    
    viz_utils = VisualizationUtils(st.session_state.solution, st.session_state.products_data, 
                                 st.session_state.resources_data, st.session_state.consumption_matrix)
    
    # Production mix chart
    st.subheader("Production Mix")
    production_chart = viz_utils.create_production_chart()
    st.plotly_chart(production_chart, use_container_width=True)
    
    # Resource utilization chart
    st.subheader("Resource Utilization")
    resource_chart = viz_utils.create_resource_utilization_chart()
    st.plotly_chart(resource_chart, use_container_width=True)
    
    # Profit breakdown
    st.subheader("Profit Contribution by Product")
    profit_chart = viz_utils.create_profit_breakdown_chart()
    st.plotly_chart(profit_chart, use_container_width=True)
    
    # Feasible region (for 2 products only)
    if len(st.session_state.products_data) == 2:
        st.subheader("Feasible Region")
        st.markdown("This chart shows the feasible region and optimal solution for the two-product case:")
        feasible_chart = viz_utils.create_feasible_region_chart()
        st.plotly_chart(feasible_chart, use_container_width=True)

def sensitivity_analysis_page():
    st.header("🔍 Sensitivity Analysis")
    
    if not st.session_state.problem_solved:
        st.warning("Please solve the problem first.")
        return
    
    analyzer = SensitivityAnalyzer(st.session_state.solver, st.session_state.products_data,
                                 st.session_state.resources_data, st.session_state.consumption_matrix)
    
    st.markdown("""
    Sensitivity analysis helps understand how changes in problem parameters affect the optimal solution.
    """)
    
    # Price sensitivity
    st.subheader("Price Sensitivity Analysis")
    
    selected_product = st.selectbox(
        "Select product for price sensitivity:",
        [p['name'] for p in st.session_state.products_data]
    )
    
    product_index = next(i for i, p in enumerate(st.session_state.products_data) if p['name'] == selected_product)
    
    col1, col2 = st.columns(2)
    
    with col1:
        price_range = st.slider(
            "Price variation range (%)",
            -50, 100, (-20, 50)
        )
    
    with col2:
        num_points = st.slider("Number of analysis points", 5, 20, 10)
    
    if st.button("Run Price Sensitivity Analysis"):
        with st.spinner("Analyzing..."):
            sensitivity_results = analyzer.price_sensitivity(product_index, price_range, num_points)
            
        if sensitivity_results:
            # Create sensitivity chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r['price'] for r in sensitivity_results],
                y=[r['profit'] for r in sensitivity_results],
                mode='lines+markers',
                name='Total Profit',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f'Price Sensitivity Analysis - {selected_product}',
                xaxis_title='Unit Price ($)',
                yaxis_title='Total Profit ($)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensitivity table
            st.subheader("Detailed Results")
            sensitivity_df = pd.DataFrame([
                {
                    'Price': f"${r['price']:.2f}",
                    'Total Profit': f"${r['profit']:.2f}",
                    'Quantity': f"{r['quantity']:.2f}",
                    'Status': r['status']
                }
                for r in sensitivity_results
            ])
            st.dataframe(sensitivity_df, use_container_width=True)
    
    # Resource capacity sensitivity
    st.subheader("Resource Capacity Sensitivity")
    
    selected_resource = st.selectbox(
        "Select resource for capacity sensitivity:",
        [r['name'] for r in st.session_state.resources_data]
    )
    
    resource_index = next(i for i, r in enumerate(st.session_state.resources_data) if r['name'] == selected_resource)
    
    col1, col2 = st.columns(2)
    
    with col1:
        capacity_range = st.slider(
            "Capacity variation range (%)",
            -30, 100, (-10, 50),
            key="capacity_range"
        )
    
    with col2:
        num_points_capacity = st.slider("Number of analysis points", 5, 20, 10, key="capacity_points")
    
    if st.button("Run Capacity Sensitivity Analysis"):
        with st.spinner("Analyzing..."):
            capacity_results = analyzer.capacity_sensitivity(resource_index, capacity_range, num_points_capacity)
            
        if capacity_results:
            # Create capacity sensitivity chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r['capacity'] for r in capacity_results],
                y=[r['profit'] for r in capacity_results],
                mode='lines+markers',
                name='Total Profit',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f'Capacity Sensitivity Analysis - {selected_resource}',
                xaxis_title='Resource Capacity',
                yaxis_title='Total Profit ($)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Capacity sensitivity table
            st.subheader("Detailed Results")
            capacity_df = pd.DataFrame([
                {
                    'Capacity': f"{r['capacity']:.2f}",
                    'Total Profit': f"${r['profit']:.2f}",
                    'Status': r['status']
                }
                for r in capacity_results
            ])
            st.dataframe(capacity_df, use_container_width=True)

def export_results_page():
    st.header("📤 Export Results")
    
    if not st.session_state.problem_solved:
        st.warning("Please solve the problem first.")
        return
    
    st.markdown("Export your optimization results and analysis:")
    
    # Prepare data for export
    solution = st.session_state.solution
    
    # Create comprehensive results dictionary
    export_data = {
        'problem_configuration': {
            'products': st.session_state.products_data,
            'resources': st.session_state.resources_data,
            'consumption_matrix': st.session_state.consumption_matrix
        },
        'solution': {
            'status': solution['status'],
            'objective_value': solution['objective_value'],
            'variables': solution['variables']
        },
        'analysis': {
            'production_plan': [],
            'resource_utilization': []
        }
    }
    
    # Add production plan
    for i, product in enumerate(st.session_state.products_data):
        var_name = f"x_{i+1}"
        quantity = solution['variables'].get(var_name, 0)
        export_data['analysis']['production_plan'].append({
            'product': product['name'],
            'quantity': quantity,
            'unit_profit': product['profit'],
            'total_revenue': quantity * product['profit']
        })
    
    # Add resource utilization
    for i, resource in enumerate(st.session_state.resources_data):
        used = sum(solution['variables'].get(f"x_{j+1}", 0) * st.session_state.consumption_matrix[i][j] 
                  for j in range(len(st.session_state.products_data)))
        export_data['analysis']['resource_utilization'].append({
            'resource': resource['name'],
            'capacity': resource['capacity'],
            'used': used,
            'utilization_percentage': (used / resource['capacity']) * 100 if resource['capacity'] > 0 else 0
        })
    
    # JSON export
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("JSON Export")
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download JSON Results",
            data=json_str,
            file_name="optimization_results.json",
            mime="application/json"
        )
    
    with col2:
        st.subheader("CSV Export")
        
        # Production plan CSV
        production_df = pd.DataFrame(export_data['analysis']['production_plan'])
        production_csv = production_df.to_csv(index=False)
        st.download_button(
            label="Download Production Plan (CSV)",
            data=production_csv,
            file_name="production_plan.csv",
            mime="text/csv"
        )
        
        # Resource utilization CSV
        resource_df = pd.DataFrame(export_data['analysis']['resource_utilization'])
        resource_csv = resource_df.to_csv(index=False)
        st.download_button(
            label="Download Resource Utilization (CSV)",
            data=resource_csv,
            file_name="resource_utilization.csv",
            mime="text/csv"
        )
    
    # Summary report
    st.subheader("Executive Summary")
    
    summary_report = f"""
# Optimization Results Summary

## Problem Overview
- **Products**: {len(st.session_state.products_data)}
- **Resources**: {len(st.session_state.resources_data)}
- **Optimization Status**: {solution['status']}

## Key Results
- **Maximum Profit**: ${solution['objective_value']:,.2f}
- **Total Units Produced**: {sum(solution['variables'].values()):,.0f}

## Production Plan
"""
    
    for item in export_data['analysis']['production_plan']:
        summary_report += f"- **{item['product']}**: {item['quantity']:.0f} units (${item['total_revenue']:,.2f} revenue)\n"
    
    summary_report += "\n## Resource Utilization\n"
    
    for item in export_data['analysis']['resource_utilization']:
        summary_report += f"- **{item['resource']}**: {item['utilization_percentage']:.1f}% utilized\n"
    
    st.download_button(
        label="Download Summary Report (Markdown)",
        data=summary_report,
        file_name="optimization_summary.md",
        mime="text/markdown"
    )
    
    # Display preview
    st.subheader("Preview")
    st.markdown(summary_report)

def database_management_page():
    st.header("🗄️ Database Management")
    
    if not st.session_state.db_initialized:
        st.error("Database not initialized properly.")
        return
    
    st.markdown("""
    Manage your saved optimization problems, solutions, and analysis results.
    """)
    
    # Database statistics
    stats = db_manager.get_problem_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Problems", stats['total_problems'])
    with col2:
        st.metric("Total Solutions", stats['total_solutions'])
    with col3:
        st.metric("Total Analyses", stats['total_analyses'])
    
    # Tabs for different database operations
    tab1, tab2, tab3, tab4 = st.tabs(["Browse Problems", "Load Problem", "Problem History", "Cleanup"])
    
    with tab1:
        st.subheader("Saved Problems")
        problems = list_problems_from_db()
        
        if problems:
            problems_df = pd.DataFrame([
                {
                    'ID': p['id'],
                    'Name': p['name'],
                    'Products': p['num_products'],
                    'Resources': p['num_resources'],
                    'Created': p['created_at'][:19] if p['created_at'] else 'N/A',
                    'Updated': p['updated_at'][:19] if p['updated_at'] else 'N/A'
                }
                for p in problems
            ])
            st.dataframe(problems_df, use_container_width=True)
        else:
            st.info("No problems saved yet. Create and save a problem from the Problem Setup page.")
    
    with tab2:
        st.subheader("Load Existing Problem")
        problems = list_problems_from_db()
        
        if problems:
            problem_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in problems}
            selected_problem = st.selectbox("Choose a problem to load:", list(problem_options.keys()))
            
            if st.button("Load Selected Problem"):
                problem_id = problem_options[selected_problem]
                problem_data = load_problem_from_db(problem_id)
                
                if problem_data:
                    # Load problem data into session state
                    st.session_state.products_data = problem_data['products_data']
                    st.session_state.resources_data = problem_data['resources_data']
                    st.session_state.consumption_matrix = problem_data['consumption_matrix']
                    st.session_state.current_problem_id = problem_id
                    
                    # Create solver with loaded data
                    st.session_state.solver = OptimizationSolver(
                        problem_data['products_data'],
                        problem_data['resources_data'],
                        problem_data['consumption_matrix']
                    )
                    
                    st.success(f"Problem '{problem_data['name']}' loaded successfully!")
                    st.info("Go to 'Mathematical Formulation' to solve the loaded problem.")
                    
                    # Show problem details
                    st.subheader("Loaded Problem Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Products:**")
                        products_df = pd.DataFrame(problem_data['products_data'])
                        st.dataframe(products_df, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Resources:**")
                        resources_df = pd.DataFrame(problem_data['resources_data'])
                        st.dataframe(resources_df, use_container_width=True)
                    
                    st.markdown("**Resource Consumption Matrix:**")
                    consumption_df = pd.DataFrame(problem_data['consumption_matrix'])
                    consumption_df.columns = [p['name'] for p in problem_data['products_data']]
                    consumption_df.index = [r['name'] for r in problem_data['resources_data']]
                    st.dataframe(consumption_df, use_container_width=True)
                else:
                    st.error("Failed to load problem data.")
        else:
            st.info("No problems available to load.")
    
    with tab3:
        st.subheader("Problem Solutions & Analysis History")
        
        if st.session_state.current_problem_id:
            problem_id = st.session_state.current_problem_id
            
            # Show solutions for current problem
            solutions = db_manager.list_solutions(problem_id)
            
            if solutions:
                st.markdown("**Solutions:**")
                solutions_df = pd.DataFrame([
                    {
                        'Solution ID': s['id'],
                        'Status': s['status'],
                        'Objective Value': f"${s['objective_value']:,.2f}" if s['objective_value'] else 'N/A',
                        'Solver Time (s)': f"{s['solver_time_seconds']:.3f}" if s['solver_time_seconds'] else 'N/A',
                        'Solved At': s['solved_at'][:19] if s['solved_at'] else 'N/A'
                    }
                    for s in solutions
                ])
                st.dataframe(solutions_df, use_container_width=True)
            else:
                st.info("No solutions found for current problem.")
            
            # Show sensitivity analyses
            analyses = db_manager.list_sensitivity_analyses(problem_id)
            
            if analyses:
                st.markdown("**Sensitivity Analyses:**")
                analyses_df = pd.DataFrame([
                    {
                        'Analysis ID': a['id'],
                        'Type': a['analysis_type'],
                        'Parameter': a['parameter_name'],
                        'Range': f"{a['range_min']:.1f} to {a['range_max']:.1f}",
                        'Points': a['num_points'],
                        'Created': a['created_at'][:19] if a['created_at'] else 'N/A'
                    }
                    for a in analyses
                ])
                st.dataframe(analyses_df, use_container_width=True)
            else:
                st.info("No sensitivity analyses found for current problem.")
        else:
            st.info("Select or create a problem to view its history.")
    
    with tab4:
        st.subheader("Database Cleanup")
        st.warning("⚠️ Cleanup operations cannot be undone!")
        
        problems = list_problems_from_db()
        if problems:
            problem_to_delete = st.selectbox(
                "Select problem to delete:",
                [""] + [f"{p['name']} (ID: {p['id']})" for p in problems]
            )
            
            if problem_to_delete and st.button("Delete Problem", type="secondary"):
                problem_id = int(problem_to_delete.split("ID: ")[1].split(")")[0])
                
                if st.checkbox("I understand this will delete all related solutions and analyses"):
                    if db_manager.delete_problem(problem_id):
                        st.success("Problem deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to delete problem.")

if __name__ == "__main__":
    main()
