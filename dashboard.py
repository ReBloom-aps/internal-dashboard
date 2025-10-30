from demand_list import create_interactive_plot_ticket, create_interactive_plot_ticket_no_listing_name, calculate_listing_ticket_metrics, calculate_demand_ticket_metrics, create_combined_demand_plot, calculate_combined_demand_ticket_metrics, calculate_ticket_size_metrics
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from bubble_api import fetch_all_pages
from bubble_api import export_institutional_users_csv, export_institutional_mom_2025
from google_api import get_website_traffic, get_country_traffic, get_search_console_data, get_active_users, get_user_types, get_country_active_users
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging 

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import logging
# import os
# logs_dir = 'logs'
# if not os.path.exists(logs_dir):
#     os.makedirs(logs_dir)
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(os.path.join(logs_dir, 'dashboard.log')),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger('dashboard')
# Set page config
st.set_page_config(
    page_title="ReBloom Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    /* Reduce st.metric value font-size and prevent truncation */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }
    </style>
""", unsafe_allow_html=True)

def create_interactive_plot(data, date_field, title, filter_func=None):
    """Create an interactive plot using Plotly"""
    # Filter data if filter function is provided
    results = data['response']['results']
    if filter_func:
        results = [item for item in results if filter_func(item)]
    
    # Extract creation dates and convert to datetime
    creation_dates = [datetime.strptime(item[date_field][:10], '%Y-%m-%d') 
                     for item in results]
    
    # Create DataFrame with dates
    df = pd.DataFrame({'Created Date': creation_dates})
    
    # Count daily creations
    daily_counts = df.groupby('Created Date').size().reset_index()
    daily_counts.columns = ['Date', 'Daily Count']
    
    # Calculate cumulative sum
    daily_counts['Cumulative Total'] = daily_counts['Daily Count'].cumsum()
    
    # Get today's date
    today = datetime.now().date()
    
    # Create a complete date range from earliest date to today
    date_range = pd.date_range(start=daily_counts['Date'].min(), end=today, freq='D')
    complete_df = pd.DataFrame({'Date': date_range})
    
    # Merge with daily_counts to fill in missing dates with 0
    daily_counts = pd.merge(complete_df, daily_counts, on='Date', how='left')
    daily_counts = daily_counts.fillna({'Daily Count': 0, 'Cumulative Total': daily_counts['Cumulative Total'].ffill()})
    
    # Calculate week-over-week growth
    latest_date = daily_counts['Date'].max()
    week_ago_date = latest_date - timedelta(days=7)
    
    # Find the closest available date for week-ago comparison
    available_dates = sorted(daily_counts['Date'])
    week_ago_date = max([d for d in available_dates if d <= week_ago_date], default=None)
    
    current_total = daily_counts['Cumulative Total'].iloc[-1]
    week_ago_total = daily_counts[daily_counts['Date'] == week_ago_date]['Cumulative Total'].iloc[0] if week_ago_date else 0
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=daily_counts['Date'],
            y=daily_counts['Daily Count'],
            name="Daily Creations",
            marker_color='#2ecc71'
        ),
        secondary_y=False
    )
    
    # Add line chart
    fig.add_trace(
        go.Scatter(
            x=daily_counts['Date'],
            y=daily_counts['Cumulative Total'],
            name="Cumulative Total",
            line=dict(color='#9b59b6', width=3)
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title} (Total: {len(results)})",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=24, color='#34495e')
        ),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=16)
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update axes with larger font sizes
    fig.update_xaxes(
        title_text="Date", 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='LightGrey',
        title_font=dict(size=18),
        tickfont=dict(size=14),
        range=[daily_counts['Date'].min(), today]
    )
    fig.update_yaxes(
        title_text="Daily Creations", 
        secondary_y=False, 
        showgrid=True, 
        gridwidth=1, 
        gridcolor='LightGrey',
        title_font=dict(size=18),
        tickfont=dict(size=14)
    )
    fig.update_yaxes(
        title_text="Cumulative Total", 
        secondary_y=True, 
        showgrid=False,
        title_font=dict(size=18),
        tickfont=dict(size=14)
    )
    
    # Add growth annotation with larger font
    if week_ago_total > 0:
        growth_percentage = ((current_total - week_ago_total) / week_ago_total) * 100
        growth_text = f"Week-over-Week Growth: +{growth_percentage:.1f}%<br>(+{current_total - week_ago_total} total)"
    else:
        if len(available_dates) > 1:
            days_of_data = (latest_date - min(available_dates)).days
            growth_text = f"Growth data limited<br>({days_of_data} days of history available)"
        else:
            growth_text = "Insufficient data for growth calculation"
    
    fig.add_annotation(
        text=growth_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.95,
        showarrow=False,
        font=dict(size=16, color='#34495e'),
        align="center"
    )
    
    return fig

def calculate_growth_metrics(data, date_field, filter_func=None):
    """Calculate growth metrics for the data"""
    results = data['response']['results']
    if filter_func:
        results = [item for item in results if filter_func(item)]
    
    # Extract creation dates and convert to datetime
    creation_dates = [datetime.strptime(item[date_field][:10], '%Y-%m-%d') 
                     for item in results]
    
    # Create DataFrame with dates
    df = pd.DataFrame({'Created Date': creation_dates})
    
    # Count daily creations
    daily_counts = df.groupby('Created Date').size().reset_index()
    daily_counts.columns = ['Date', 'Daily Count']
    
    # Calculate cumulative sum
    daily_counts['Cumulative Total'] = daily_counts['Daily Count'].cumsum()
    # Calculate week-over-week growth
    latest_date = pd.to_datetime(daily_counts['Date'].max())
    week_ago_date = latest_date - timedelta(days=7)
    # Find the closest available date for week-ago comparison
    available_dates = sorted(daily_counts['Date'])
    week_ago_date = max([d for d in available_dates if d <= week_ago_date], default=None)
    
    current_total = daily_counts['Cumulative Total'].iloc[-1]
    week_ago_total = daily_counts[daily_counts['Date'] == week_ago_date]['Cumulative Total'].iloc[0] if week_ago_date else 0
    # Calculate weekly addition
    weekly_addition = current_total - week_ago_total if week_ago_total > 0 else current_total
    
    # Calculate growth percentage
    if week_ago_total > 0:
        growth_percentage = ((current_total - week_ago_total) / week_ago_total) * 100
    else:
        growth_percentage = None
    
    return current_total, weekly_addition, growth_percentage
def get_min_ticket_size_value(number):
    """Get the min ticket size value from a ticket size string"""
    if number == 1:
        return 0.5
    elif number == 5:
        return 1
    elif number == 10:
        return 5
    elif number == 20:
        return 10
    elif number == 50:
        return 20
    elif number == 100:
        return 50
    else:
        return 0
    
    ticket_size_str = str(ticket_size_str).strip()
def _parse_iso_date(date_value):
    """Parse an ISO-like string (e.g., 2024-11-07T08:07:28.307Z or 2024-11-07) to datetime.date.
    Returns None if parsing fails."""
    if not date_value:
        return None
    try:
        date_part = date_value[:10] if isinstance(date_value, str) and 'T' in date_value else date_value
        return datetime.strptime(date_part, '%Y-%m-%d')
    except Exception:
        return None

def parse_single_ticket_size_max(ticket_size_str):
    """Parse a single ticket size string and return value in millions (same logic as demand_list.py)"""
    if not ticket_size_str or ticket_size_str == '':
        return 0
    
    ticket_size_str = str(ticket_size_str).strip()
    
    if '<$1m' in ticket_size_str:
        return 1  
    elif '$1-5m' in ticket_size_str:
        return 5
    elif '$5-10m' in ticket_size_str:
        return 10
    elif '$10-20m' in ticket_size_str:
        return 20
    elif '$20-50m' in ticket_size_str:
        return 50
    elif '$50-100m' in ticket_size_str:
        return 100
    elif '$100m+' in ticket_size_str:
        return 100  # Assume 100m for 100m+
    elif 'Not specified' in ticket_size_str:
        return 0
    elif 'All' in ticket_size_str:
        return 100
    else:
        return 0
def parse_single_ticket_size_min(ticket_size_str):
    """Parse a single ticket size string and return value in millions (same logic as demand_list.py)"""
    if not ticket_size_str or ticket_size_str == '':
        return 0
    
    ticket_size_str = str(ticket_size_str).strip()
    
    if '<$1m' in ticket_size_str:
        return 0.5
    elif '$1-5m' in ticket_size_str:
        return 1
    elif '$5-10m' in ticket_size_str:
        return 5
    elif '$10-20m' in ticket_size_str:
        return 10
    elif '$20-50m' in ticket_size_str:
        return 20

    elif '$50-100m' in ticket_size_str:
        return 50
    elif '$100m+' in ticket_size_str:
        return 100
    elif 'Not specified' in ticket_size_str:
        return 0
    elif 'All' in ticket_size_str:
        return 0.5
    else:
        return 0

def add_highest_ticket_size(endpoint_data_investor_preference):
    """
    Create a new dataset with highest_ticket_size field from the ticket_sizes list
    """
    import copy
    
    if not endpoint_data_investor_preference or 'response' not in endpoint_data_investor_preference:
        return endpoint_data_investor_preference
    
    # Create a deep copy to avoid modifying the original
    investor_preference_with_highest_lowest = copy.deepcopy(endpoint_data_investor_preference)
    
    # Process each item in the results
    for item in investor_preference_with_highest_lowest['response']['results']:
        ticket_sizes_list = item.get('ticket_sizes', [])
        
        if ticket_sizes_list and isinstance(ticket_sizes_list, list):
            # Parse each ticket size and find the min/max
            parsed_sizes = []
            for ticket_size in ticket_sizes_list:
                parsed_value = parse_single_ticket_size_max(ticket_size)
                parsed_sizes.append(parsed_value)
            
            # Find the highest and lowest ticket sizes
            highest_size = max(parsed_sizes) if parsed_sizes else 0
            lowest_size = get_min_ticket_size_value(min(parsed_sizes)) if parsed_sizes else 0
            
            # Find the original string that corresponds to the highest value
            highest_ticket_string = ""
            for ticket_size in ticket_sizes_list:
                if parse_single_ticket_size_max(ticket_size) == highest_size:
                    highest_ticket_string = ticket_size
                    break
            
            # Add the new fields
            item['highest_ticket_size'] = highest_ticket_string
            item['highest_ticket_size_value'] = highest_size
            item['lowest_ticket_size_value'] = lowest_size
            
            print(f"Processed ticket_sizes {ticket_sizes_list} -> highest: {highest_ticket_string} (${highest_size}M)")
        else:
            # No ticket_sizes or empty list
            item['highest_ticket_size'] = "Not specified"
            item['highest_ticket_size_value'] = 0
            item['lowest_ticket_size_value'] = 0
            print(f"No ticket_sizes found, setting to 'Not specified'")
    
    print(f"=== HIGHEST TICKET SIZE PROCESSING COMPLETE ===")
    print(f"Processed {len(investor_preference_with_highest_lowest['response']['results'])} investor preference records")
    
    return investor_preference_with_highest_lowest

def add_listing_name(endpoint_data_listing, endpoint_data_deal_specification):
    # create a new endpoint data to replace the 'OG listing' in the deal specification which is the same as 'company' in the listing endpoint with the corresponding company_name'
    import copy
    
    # Debug: Print the structure of the listing data to understand the field names
    print("=== DEBUGGING LISTING DATA ===")
    if endpoint_data_listing and 'response' in endpoint_data_listing and 'results' in endpoint_data_listing['response']:
        sample_listing = endpoint_data_listing['response']['results'][0] if endpoint_data_listing['response']['results'] else {}
        print("Sample listing fields:", list(sample_listing.keys()))
        print("First few listings:")
        for i, item in enumerate(endpoint_data_listing['response']['results'][:3]):
            print(f"Listing {i}: _id={item.get('_id')}, company={item.get('company')}, company_name={item.get('company_name')}")
    
    # Debug: Print the structure of the deal specification data
    print("=== DEBUGGING DEAL SPECIFICATION DATA ===")
    if endpoint_data_deal_specification and 'response' in endpoint_data_deal_specification and 'results' in endpoint_data_deal_specification['response']:
        sample_deal = endpoint_data_deal_specification['response']['results'][0] if endpoint_data_deal_specification['response']['results'] else {}
        print("Sample deal specification fields:", list(sample_deal.keys()))
        print("First few deal specifications:")
        for i, item in enumerate(endpoint_data_deal_specification['response']['results'][:3]):
            print(f"Deal {i}: _id={item.get('_id')}, OG listing={item.get('OG listing')}")
    
    # Create a mapping dictionary from company ID to company name
    company_mapping = {}
    
    # Try multiple possible field names for company ID and name
    possible_id_fields = ['_id', 'company', 'company_id']
    possible_name_fields = ['company_name', 'name', 'Listing ⁠Name']
    
    for item in endpoint_data_listing['response']['results']:
        company_id = None
        company_name = None
        
        # Try to find company ID
        for id_field in possible_id_fields:
            if item.get(id_field):
                company_id = item.get(id_field)
                break
        
        # Try to find company name
        for name_field in possible_name_fields:
            if item.get(name_field):
                company_name = item.get(name_field)
                break
        
        if company_id and company_name:
            company_mapping[company_id] = company_name
    
   
    # Create a deep copy of the deal specification data to avoid modifying the original
    deal_specification_with_company_name = copy.deepcopy(endpoint_data_deal_specification)
    
    # Now update the copied deal specification data
    matched_count = 0
    total_count = len(deal_specification_with_company_name['response']['results'])
    
    for item in deal_specification_with_company_name['response']['results']:
        og_listing_id = item.get('OG listing')  # This should match company IDs from listing
        if og_listing_id:
            if og_listing_id in company_mapping:
                # Replace the ID with the readable company name
                item['Listing Name'] = company_mapping[og_listing_id]
                item['Listing ID'] = og_listing_id
                matched_count += 1
            else:
                # Set a default value for unmatched listings
                item['Listing Name'] = f"Unknown ({og_listing_id})"
                item['Listing ID'] = og_listing_id
        else:
            item['Listing Name'] = "No OG listing specified"
            item['Listing ID'] = None
    
    return deal_specification_with_company_name

def main():
    st.title("📊 ReBloom Analytics Dashboard")
    
    # Add refresh button
    if st.button("🔄 Refresh Data"):
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            st.rerun()
    
    # Fetch data
    with st.spinner("Fetching data from all sources..."):
        # Fetch Bubble API data
        # Deal specification needs be after Listing, because it uses the Listing endpoint to get the company name
        endpoints = ['Listing', 'User', 'Match', 'Demand_listing', 'Deal specification', 'Investor preference']
        endpoint_data = {}
        
        for endpoint in endpoints:
            data = fetch_all_pages(endpoint)
            if data:
                endpoint_data[endpoint] = data
        
        endpoint_data['Deal specification with company name'] = add_listing_name(endpoint_data['Listing'], endpoint_data['Deal specification'])
        
        # Process investor preference data to add highest ticket size
        if 'Investor preference' in endpoint_data:
            endpoint_data['Investor preference with min max'] = add_highest_ticket_size(endpoint_data['Investor preference'])

        
        # Write exports for institutional users
        if 'User' in endpoint_data:
            import os
            os.makedirs('exports', exist_ok=True)
            csv_users = os.path.join('exports', 'institutional_users.csv')
            csv_mom = os.path.join('exports', 'institutional_signups_2025_mom.csv')
            png_mom = os.path.join('exports', 'institutional_signups_2025_mom.png')
            try:
                export_institutional_users_csv(endpoint_data['User'], csv_users)
                export_institutional_mom_2025(endpoint_data['User'], 'Created Date', csv_mom, png_mom)
            except Exception as e:
                st.warning(f"Could not write exports: {e}")

        # Fetch Google Analytics data
        website_traffic = get_website_traffic()
        country_traffic = get_country_traffic()
        search_console_data = get_search_console_data()
        active_users_data = get_active_users()
        user_types_data = get_user_types()
        country_active_users = get_country_active_users()
    
    # Display metrics in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Platform Metrics", "Website Analytics", "Search Performance", "Supply & Demand"])
    
    with tab1:
        # Display Bubble API metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Listing' in endpoint_data:
                total, weekly_add, growth = calculate_growth_metrics(endpoint_data['Listing'], 'Created Date')
                growth_text = f"+{growth:.1f}%" if growth is not None else "N/A"
                st.metric(
                    "Total Listings",
                    f"{total} (WoW growth: +{weekly_add}, {growth_text})",
                    help="Total number of listings in the system"
                )
        
        with col2:
            if 'User' in endpoint_data:
                total, weekly_add, growth = calculate_growth_metrics(endpoint_data['User'], 'Created Date')
                growth_text = f"+{growth:.1f}%" if growth is not None else "N/A"
                st.metric(
                    "Total Users",
                    f"{total} (WoW growth: +{weekly_add}, {growth_text})",
                    help="Total number of users in the system"
                )
        
        with col3:
            if 'Match' in endpoint_data:
                total, weekly_add, growth = calculate_growth_metrics(
                    endpoint_data['Match'], 
                    'Created Date',
                    filter_func=lambda x: x.get('seller_match_status') == 'Active'
                )
                growth_text = f"+{growth:.1f}%" if growth is not None else "N/A"
                st.metric(
                    "Active Matches",
                    f"{total} (WoW growth: +{weekly_add}, {growth_text})",
                    help="Total number of active matches"
                )
        
        # Display Bubble API plots
        if 'Listing' in endpoint_data:
            st.plotly_chart(
                create_interactive_plot(
                    endpoint_data['Listing'],
                    'Created Date',
                    'Listing Creation Timeline'
                ),
                use_container_width=True
            )
        
        if 'User' in endpoint_data:
            st.plotly_chart(
                create_interactive_plot(
                    endpoint_data['User'],
                    'Created Date',
                    'User Growth Timeline'
                ),
                use_container_width=True
            )
        
        if 'Match' in endpoint_data:
            st.plotly_chart(
                create_interactive_plot(
                    endpoint_data['Match'],
                    'Created Date',
                    'Active Match Timeline',
                    filter_func=lambda x: x.get('seller_match_status') == 'Active'
                ),
                use_container_width=True
            )
    
    with tab2:
        # Display Google Analytics metrics
        if active_users_data is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_active_users = active_users_data['active_users'].sum()
                st.metric("Active Users", total_active_users)
            
            if user_types_data is not None:
                with col2:
                    new_users = user_types_data[user_types_data['user_type'] == 'new']['active_users'].sum()
                    st.metric("New Users", new_users)
            
            with col3:
                avg_engagement_time = active_users_data['active_users'].mean()  # Convert to minutes
                st.metric("Avg. Engagement Time per Active User", f"{avg_engagement_time:.1f} seconds")
        
        # Display active users trend
        if active_users_data is not None:
            st.subheader("Active Users Trends")
            fig = px.line(
                active_users_data.sort_values('date'),
                x='date', 
                y='active_users',
                title='Active Users Trends',
                labels={'active_users': 'Active Users', 'date': 'Date'},
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display active users by country
        if country_active_users is not None:
            st.subheader("Active Users by Country")
            
            # Create two columns for map and table
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.choropleth(country_active_users,
                                  locations='country',
                                  locationmode='country names',
                                  color='active_users',
                                  hover_name='country',
                                  title='Active Users by Country',
                                  template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sort countries by active users in descending order
                country_table = country_active_users.sort_values('active_users', ascending=False)
                # Format the table
                country_table = country_table.rename(columns={
                    'country': 'Country',
                    'active_users': 'Active Users'
                })
                st.dataframe(
                    country_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Country": st.column_config.TextColumn("Country"),
                        "Active Users": st.column_config.NumberColumn("Active Users", format="%d")
                    }
                )
    
    with tab3:
        # Display Search Console metrics
        if search_console_data is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_clicks = search_console_data['clicks'].sum()
                st.metric("Total Clicks", total_clicks)
            
            with col2:
                total_impressions = search_console_data['impressions'].sum()
                st.metric("Total Impressions", total_impressions)
            
            with col3:
                avg_ctr = search_console_data['ctr'].mean() * 100
                st.metric("Average CTR", f"{avg_ctr:.1f}%")
            
            with col4:
                avg_position = search_console_data['position'].mean()
                st.metric("Average Position", f"{avg_position:.1f}")
            
            # Plot clicks over the past 28 days
            st.subheader("Clicks Over the Past 28 Days")
            clicks_by_date = search_console_data.groupby('query').agg({'clicks': 'sum'}).reset_index()
            fig = px.bar(
                clicks_by_date,
                x='query',
                y='clicks',
                title='Clicks by Query',
                labels={'clicks': 'Clicks', 'query': 'Query'},
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top queries
            st.subheader("Top Search Queries")
            top_queries = search_console_data.sort_values('clicks', ascending=False).head(10)
            fig = px.bar(top_queries, x='query', y=['clicks', 'impressions'],
                        title='Top Search Queries',
                        labels={'value': 'Count', 'query': 'Search Query'},
                        template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # Device distribution
            st.subheader("Traffic by Device")
            device_data = search_console_data.groupby('device').agg({
                'clicks': 'sum',
                'impressions': 'sum'
            }).reset_index()
            fig = px.pie(device_data, values='clicks', names='device',
                        title='Clicks by Device',
                        template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Display ticket size metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Deal specification with company name' in endpoint_data:
                total_min, total_max, listing_count, avg_min, avg_max = calculate_listing_ticket_metrics(endpoint_data['Deal specification with company name'])
                mean_total = (total_min + total_max) / 2
                st.metric(
                    "Total Supply Ticket Size",
                    f"${total_min:.1f}M - ${total_max:.1f}M",
                    delta=f"Mean: ${mean_total:.1f}M",
                    help="Total accumulated ticket size range of all platform listings (minimum to maximum with mean)"
                )
                st.metric(
                    "Total Number of Supply",
                    f"{listing_count}",
                    help="Total number of listings with specified ticket sizes"
                )

               
        
        with col2:
            if 'Deal specification with company name' in endpoint_data:
                total_min, total_max, listing_count, avg_min, avg_max = calculate_listing_ticket_metrics(endpoint_data['Deal specification with company name'])
                mean_total = (total_min + total_max) / 2
                 # New: Darkpool Listing Ticket Size (only listings where corresponding Listing.dark pool? is True)
                try:
                    ds = endpoint_data['Deal specification with company name']
                    ls = endpoint_data.get('Listing')
                    if ls and 'response' in ls and 'results' in ls['response']:
                        darkpool_ds = (
                            lambda ds_in, ls_in: {
                                'response': {
                                    'results': [
                                        item for item in ds_in['response']['results']
                                        if ({r.get('_id'): r.get('dark pool?') for r in ls_in['response']['results']}.get(item.get('OG listing')) is True)
                                    ]
                                }
                            }
                        )(ds, ls)
                        dp_min, dp_max, dp_count, dp_avg_min, dp_avg_max = calculate_listing_ticket_metrics(darkpool_ds)
                        dp_mean = (dp_min + dp_max) / 2
                        st.metric(
                            "Darkpool Supply Ticket Size",
                            f"${dp_min:.1f}M - ${dp_max:.1f}M",
                            delta=f"Mean: ${dp_mean:.1f}M",
                            help="Ticket size range for listings whose Listing.dark pool? is True"
                        )

                        # Public Listing Ticket Size = Total Listing Ticket Size - Darkpool Listing Ticket Size
                        public_min = max(total_min - dp_min, 0)
                        public_max = max(total_max - dp_max, 0)
                        public_mean = (public_min + public_max) / 2
                        st.metric(
                            "Public Supply Ticket Size",
                            f"${public_min:.1f}M - ${public_max:.1f}M",
                            delta=f"Mean: ${public_mean:.1f}M",
                            help="Ticket size range for listings whose Listing.dark pool? is False or not set"
                        )
                except Exception:
                    pass
        
        with col3:
            if 'Investor preference with min max' in endpoint_data and 'Demand_listing' in endpoint_data:
                total_min_combined, total_max_combined, count_combined = calculate_combined_demand_ticket_metrics(
                    endpoint_data['Investor preference with min max'],
                    endpoint_data['Demand_listing']
                )
                mean_combined = (total_min_combined + total_max_combined) / 2
                st.metric(
                    "Total Demand Ticket Size",
                    f"${total_min_combined:.1f}M - ${total_max_combined:.1f}M",
                    delta=f"Mean: ${mean_combined:.1f}M",
                    help="Total accumulated ticket size range of all demand side preferences and listings (minimum to maximum with mean)"
                )

            if 'Investor preference with min max' in endpoint_data and 'Demand_listing' in endpoint_data:
                st.metric(
                    "Total Number of Demands",
                    f"{count_combined}",
                    help="Total number of demand entries (investor preferences + demand listings) with specified ticket sizes"
                )
        
        with col4:
            # New metric: Public Demand Ticket Size (Investor preference with min max only)
            if 'Investor preference with min max' in endpoint_data:
                total_min_pub = 0
                total_max_pub = 0
                count_pub = 0
                try:
                    investor_preference_data = endpoint_data['Investor preference with min max']
                    if investor_preference_data and 'response' in investor_preference_data and 'results' in investor_preference_data['response']:
                        for item in investor_preference_data['response']['results']:
                            try:
                                size_str = item.get('highest_ticket_size', '')
                                min_val = item.get('lowest_ticket_size_value', 0)
                                if size_str:
                                    parsed_max = parse_single_ticket_size_max(size_str)
                                    if parsed_max > 0:
                                        total_max_pub += parsed_max
                                        # Ensure numeric min value
                                        try:
                                            min_num = float(min_val)
                                        except (ValueError, TypeError):
                                            min_num = 0
                                        total_min_pub += min_num
                                        count_pub += 1
                            except (ValueError, TypeError):
                                continue
                except Exception:
                    pass

                mean_pub = (total_min_pub + total_max_pub) / 2 if (total_min_pub + total_max_pub) > 0 else 0
                st.metric(
                    "Public Demand Ticket Size",
                    f"${total_min_pub:.1f}M - ${total_max_pub:.1f}M",
                    delta=f"Mean: ${mean_pub:.1f}M",
                    help="Ticket size totals computed from Investor Preference only"
                )

            # New metric: Darkpool Demand Ticket Size (Demand_listing only)
            if 'Demand_listing' in endpoint_data:
                total_min_dl, total_max_dl, count_dl, avg_min_dl, avg_max_dl = calculate_ticket_size_metrics(
                    endpoint_data['Demand_listing'],
                    modified_date='Modified Date',
                    ticket_size='Ticketsize',
                )
                mean_dl = (total_min_dl + total_max_dl) / 2
                st.metric(
                    "Darkpool Demand Ticket Size",
                    f"${total_min_dl:.1f}M - ${total_max_dl:.1f}M",
                    delta=f"Mean: ${mean_dl:.1f}M",
                    help="Ticket size totals computed from Demand Listings only"
                )
        
        # Today section: duplicate the same eight metrics below without any filtering
        st.markdown(
            """
            <style>
            .today-box { background: #e8f5e9; padding: 16px; border-radius: 10px; }
            .archived-box { background: #f2f3f5; padding: 16px; border-radius: 10px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="today-box">', unsafe_allow_html=True)
        st.subheader("Active")
        
        col1_t, col2_t, col3_t, col4_t = st.columns(4)

        with col1_t:
            if 'Deal specification with company name' in endpoint_data:
                total_min, total_max, listing_count, avg_min, avg_max = calculate_listing_ticket_metrics(endpoint_data['Deal specification with company name'], status='Active')
                mean_total = (total_min + total_max) / 2
                st.metric(
                    "Total Supply Ticket Size",
                    f"${total_min:.1f}M - ${total_max:.1f}M",
                    delta=f"Mean: ${mean_total:.1f}M",
                    help="Total accumulated ticket size range of all platform listings (minimum to maximum with mean)"
                )
                st.metric(
                    "Total Number of Supply",
                    f"{listing_count}",
                    help="Total number of listings with specified ticket sizes"
                )

        with col2_t:
            if 'Deal specification with company name' in endpoint_data:
                total_min, total_max, listing_count, avg_min, avg_max = calculate_listing_ticket_metrics(endpoint_data['Deal specification with company name'], status='Active')
                mean_total = (total_min + total_max) / 2
                # Darkpool Listing Ticket Size (only listings where corresponding Listing.dark pool? is True)
                try:
                    ds = endpoint_data['Deal specification with company name']
                    ls = endpoint_data.get('Listing')
                    if ls and 'response' in ls and 'results' in ls['response']:
                        # Filter listings to only Active status
                        _ls_results_active = [r for r in ls['response']['results'] if r.get('status') == 'Active']
                        darkpool_ds = (
                            lambda ds_in, ls_in: {
                                'response': {
                                    'results': [
                                        item for item in ds_in['response']['results']
                                        if ({r.get('_id'): r.get('dark pool?') for r in ls_in}.get(item.get('OG listing')) is True)
                                    ]
                                }
                            }
                        )(ds, _ls_results_active)
                        dp_min, dp_max, dp_count, dp_avg_min, dp_avg_max = calculate_listing_ticket_metrics(darkpool_ds, status='Active')
                        dp_mean = (dp_min + dp_max) / 2
                        st.metric(
                            "Darkpool Supply Ticket Size",
                            f"${dp_min:.1f}M - ${dp_max:.1f}M",
                            delta=f"Mean: ${dp_mean:.1f}M",
                            help="Ticket size range for listings whose Listing.dark pool? is True"
                        )

                        # Public Listing Ticket Size = Total Listing Ticket Size - Darkpool Listing Ticket Size
                        public_min = max(total_min - dp_min, 0)
                        public_max = max(total_max - dp_max, 0)
                        public_mean = (public_min + public_max) / 2
                        st.metric(
                            "Public Supply Ticket Size",
                            f"${public_min:.1f}M - ${public_max:.1f}M",
                            delta=f"Mean: ${public_mean:.1f}M",
                            help="Ticket size range for listings whose Listing.dark pool? is False or not set"
                        )
                except Exception:
                    pass

        with col3_t:
            if 'Investor preference with min max' in endpoint_data and 'Demand_listing' in endpoint_data:
                total_min_combined, total_max_combined, count_combined = calculate_combined_demand_ticket_metrics(
                    endpoint_data['Investor preference with min max'],
                    endpoint_data['Demand_listing'],
                    status='Active'
                )
                mean_combined = (total_min_combined + total_max_combined) / 2
                st.metric(
                    "Total Demand Ticket Size",
                    f"${total_min_combined:.1f}M - ${total_max_combined:.1f}M",
                    delta=f"Mean: ${mean_combined:.1f}M",
                    help="Total accumulated ticket size range of all demand side preferences and listings (minimum to maximum with mean)"
                )

            if 'Investor preference with min max' in endpoint_data and 'Demand_listing' in endpoint_data:
                st.metric(
                    "Total Number of Demands",
                    f"{count_combined}",
                    help="Total number of demand entries (investor preferences + demand listings) with specified ticket sizes"
                )

        with col4_t:
            # Public Demand Ticket Size (Investor preference with min max only)
            if 'Investor preference with min max' in endpoint_data:
                total_min_pub = 0
                total_max_pub = 0
                count_pub = 0
                try:
                    investor_preference_data = endpoint_data['Investor preference with min max']
                    if investor_preference_data and 'response' in investor_preference_data and 'results' in investor_preference_data['response']:
                        cutoff_dt = datetime.now() - timedelta(days=90)
                        investor_preference_active = [
                            r for r in investor_preference_data['response']['results']
                            if (d := _parse_iso_date(r.get('Modified Date'))) is not None and d < cutoff_dt
                        ]
                        for item in investor_preference_active:
                            try:
                                size_str = item.get('highest_ticket_size', '')
                                min_val = item.get('lowest_ticket_size_value', 0)
                                if size_str:
                                    parsed_max = parse_single_ticket_size_max(size_str)
                                    if parsed_max > 0:
                                        total_max_pub += parsed_max
                                        # Ensure numeric min value
                                        try:
                                            min_num = float(min_val)
                                        except (ValueError, TypeError):
                                            min_num = 0
                                        total_min_pub += min_num
                                        count_pub += 1
                            except (ValueError, TypeError):
                                continue
                except Exception:
                    pass

                mean_pub = (total_min_pub + total_max_pub) / 2 if (total_min_pub + total_max_pub) > 0 else 0
                st.metric(
                    "Public Demand Ticket Size",
                    f"${total_min_pub:.1f}M - ${total_max_pub:.1f}M",
                    delta=f"Mean: ${mean_pub:.1f}M",
                    help="Ticket size totals computed from Investor Preference only"
                )

            # Darkpool Demand Ticket Size (Demand_listing only)
            if 'Demand_listing' in endpoint_data:
                total_min_dl, total_max_dl, count_dl, avg_min_dl, avg_max_dl = calculate_ticket_size_metrics(
                    endpoint_data['Demand_listing'],
                    modified_date='Modified Date',
                    ticket_size='Ticketsize',
                    status='Active'
                )
                mean_dl = (total_min_dl + total_max_dl) / 2
                st.metric(
                    "Darkpool Demand Ticket Size",
                    f"${total_min_dl:.1f}M - ${total_max_dl:.1f}M",
                    delta=f"Mean: ${mean_dl:.1f}M",
                    help="Ticket size totals computed from Demand Listings only"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Archived section in an expander and grey background
        with st.expander("Archived", expanded=False):
            st.markdown('<div class="archived-box">', unsafe_allow_html=True)
            
            col1_t, col2_t, col3_t, col4_t = st.columns(4)

        with col1_t:
            if 'Deal specification with company name' in endpoint_data:
                total_min, total_max, listing_count, avg_min, avg_max = calculate_listing_ticket_metrics(endpoint_data['Deal specification with company name'],status='Archived')
                mean_total = (total_min + total_max) / 2
                st.metric(
                    "Total Supply Ticket Size",
                    f"${total_min:.1f}M - ${total_max:.1f}M",
                    delta=f"Mean: ${mean_total:.1f}M",
                    help="Total accumulated ticket size range of all platform listings (minimum to maximum with mean)"
                )
                st.metric(
                    "Total Number of Supply",
                    f"{listing_count}",
                    help="Total number of listings with specified ticket sizes"
                )

        with col2_t:
            if 'Deal specification with company name' in endpoint_data:
                total_min, total_max, listing_count, avg_min, avg_max = calculate_listing_ticket_metrics(endpoint_data['Deal specification with company name'],status='Archived')
                mean_total = (total_min + total_max) / 2
                # Darkpool Listing Ticket Size (only listings where corresponding Listing.dark pool? is True)
                try:
                    ds = endpoint_data['Deal specification with company name']
                    ls = endpoint_data.get('Listing')
                    if ls and 'response' in ls and 'results' in ls['response']:
                        darkpool_ds = (
                            lambda ds_in, ls_in: {
                                'response': {
                                    'results': [
                                        item for item in ds_in['response']['results']
                                        if ({r.get('_id'): r.get('dark pool?') for r in ls_in['response']['results']}.get(item.get('OG listing')) is True)
                                    ]
                                }
                            }
                        )(ds, ls)
                        dp_min, dp_max, dp_count, dp_avg_min, dp_avg_max = calculate_listing_ticket_metrics(darkpool_ds, status='Archived')
                        dp_mean = (dp_min + dp_max) / 2
                        st.metric(
                            "Darkpool Supply Ticket Size",
                            f"${dp_min:.1f}M - ${dp_max:.1f}M",
                            delta=f"Mean: ${dp_mean:.1f}M",
                            help="Ticket size range for listings whose Listing.dark pool? is True"
                        )

                        # Public Listing Ticket Size = Total Listing Ticket Size - Darkpool Listing Ticket Size
                        public_min = max(total_min - dp_min, 0)
                        public_max = max(total_max - dp_max, 0)
                        public_mean = (public_min + public_max) / 2
                        st.metric(
                            "Public Supply Ticket Size",
                            f"${public_min:.1f}M - ${public_max:.1f}M",
                            delta=f"Mean: ${public_mean:.1f}M",
                            help="Ticket size range for listings whose Listing.dark pool? is False or not set"
                        )
                except Exception:
                    pass

        with col3_t:
            if 'Investor preference with min max' in endpoint_data and 'Demand_listing' in endpoint_data:
                total_min_combined, total_max_combined, count_combined = calculate_combined_demand_ticket_metrics(
                    endpoint_data['Investor preference with min max'],
                    endpoint_data['Demand_listing'],
                    status='Archived'
                )
                mean_combined = (total_min_combined + total_max_combined) / 2
                st.metric(
                    "Total Demand Ticket Size",
                    f"${total_min_combined:.1f}M - ${total_max_combined:.1f}M",
                    delta=f"Mean: ${mean_combined:.1f}M",
                    help="Total accumulated ticket size range of all demand side preferences and listings (minimum to maximum with mean)"
                )

            if 'Investor preference with min max' in endpoint_data and 'Demand_listing' in endpoint_data:
                st.metric(
                    "Total Number of Demands",
                    f"{count_combined}",
                    help="Total number of demand entries (investor preferences + demand listings) with specified ticket sizes"
                )

        with col4_t:
            # Public Demand Ticket Size (Investor preference with min max only)
            if 'Investor preference with min max' in endpoint_data:
                total_min_pub = 0
                total_max_pub = 0
                count_pub = 0
                try:
                    investor_preference_data = endpoint_data['Investor preference with min max']
                    if investor_preference_data and 'response' in investor_preference_data and 'results' in investor_preference_data['response']:
                        cutoff_dt = datetime.now() - timedelta(days=90)
                        investor_preference_archived = [
                            r for r in investor_preference_data['response']['results']
                            if (d := _parse_iso_date(r.get('Modified Date'))) is not None and d >= cutoff_dt
                        ]
                        for item in investor_preference_archived:
                            try:
                                size_str = item.get('highest_ticket_size', '')
                                min_val = item.get('lowest_ticket_size_value', 0)
                                if size_str:
                                    parsed_max = parse_single_ticket_size_max(size_str)
                                    if parsed_max > 0:
                                        total_max_pub += parsed_max
                                        # Ensure numeric min value
                                        try:
                                            min_num = float(min_val)
                                        except (ValueError, TypeError):
                                            min_num = 0
                                        total_min_pub += min_num
                                        count_pub += 1
                            except (ValueError, TypeError):
                                continue
                except Exception:
                    pass

                mean_pub = (total_min_pub + total_max_pub) / 2 if (total_min_pub + total_max_pub) > 0 else 0
                st.metric(
                    "Public Demand Ticket Size",
                    f"${total_min_pub:.1f}M - ${total_max_pub:.1f}M",
                    delta=f"Mean: ${mean_pub:.1f}M",
                    help="Ticket size totals computed from Investor Preference only"
                )

            # Darkpool Demand Ticket Size (Demand_listing only)
            if 'Demand_listing' in endpoint_data:
                total_min_dl, total_max_dl, count_dl, avg_min_dl, avg_max_dl = calculate_ticket_size_metrics(
                    endpoint_data['Demand_listing'],
                    modified_date='Modified Date',
                    ticket_size='Ticketsize',
                    status='Archived'
                )
                mean_dl = (total_min_dl + total_max_dl) / 2
                st.metric(
                    "Darkpool Demand Ticket Size",
                    f"${total_min_dl:.1f}M - ${total_max_dl:.1f}M",
                    delta=f"Mean: ${mean_dl:.1f}M",
                    help="Ticket size totals computed from Demand Listings only"
                )
            
        
        # Add clarification note about ticket size calculations


        if 'Listing' in endpoint_data and 'Deal specification with company name' in endpoint_data:

            def is_darkpool(item):
                return item.get("dark pool?") is True
            
            st.subheader("DarkPool Supply")
            st.plotly_chart(
                create_interactive_plot_ticket(
                    endpoint_data['Listing'],
                    'Darkpool Supply Creation Timeline',
                    modified_date = 'Modified Date',
                    ticket_size = 'Ticket size',
                    listing_name = 'company_name',
                    second_response_data = endpoint_data['Deal specification with company name'],
                    filter_func=is_darkpool
                ),
                use_container_width=True
            )
        else:
            st.write("No deal specification data found")
        
        if 'Deal specification with company name' in endpoint_data:
            st.subheader("Public Supply")
            st.plotly_chart(
                create_interactive_plot_ticket(
                    # Join Deal spec with Listing on OG listing (deal) -> _id (listing) and filter out dark pool
                    (lambda ds, ls: {
                        'response': {
                            'results': [
                                item for item in ds['response']['results']
                                if not ({r.get('_id'): r.get('dark pool?') for r in ls['response']['results']}.get(item.get('OG listing')) is True)
                            ]
                        }
                    })(endpoint_data['Deal specification with company name'], endpoint_data['Listing']),
                    'Public Supply Creation Timeline',
                    modified_date = 'Modified Date',
                    ticket_size = 'ticket_size',
                    listing_name = 'Listing Name',
                ),
                use_container_width=True
            )
        else:
            st.write("No deal specification data found")

        # if 'Investor preference with min max' in endpoint_data:
        #     st.subheader("Demand Side Ticket Size")

        #     st.plotly_chart(
        #         create_interactive_plot_ticket_no_listing_name(
        #             endpoint_data['Investor preference with min max'],
        #             'Investor Preference Creation Timeline',
        #             modified_date = 'Modified Date',
        #             ticket_size = 'highest_ticket_size',
        #         ),
        #         use_container_width=True
        #     )
        # else:
        #     st.write("No investor preference data found")

        # if 'Demand_listing' in endpoint_data:
        #     st.subheader("Dark Pool Ticket Size")
            
        #     st.plotly_chart(
        #         create_interactive_plot_ticket(
        #             endpoint_data['Demand_listing'],
        #             'Demand Listing Creation Timeline'
        #         ),
        #         use_container_width=True
        #     )
        # else:
        #     st.write("No demand listing data found")

        # Combined demand plot
        if 'Investor preference with min max' in endpoint_data and 'Demand_listing' in endpoint_data:
            st.subheader("Demand for Public & Dark Pool Listings")
            
            st.plotly_chart(
                create_combined_demand_plot(
                    endpoint_data['Investor preference with min max'],
                    endpoint_data['Demand_listing'],
                    'Demand Creation Timeline',
                    modified_date='Modified Date'
                ),
                use_container_width=True
            )
        else:
            st.write("Combined demand plot requires both investor preference and demand listing data")
        st.info("📝 **Note**: When user selects 'Not specified' as the ticket size, it is treated as $0M in all calculations above.")

    # Add timestamp
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 