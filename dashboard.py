import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from bubble_api import fetch_all_pages
from google_api import get_website_traffic, get_country_traffic, get_search_console_data, get_active_users, get_user_types, get_country_active_users
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    latest_date = daily_counts['Date'].max()
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

def main():
    st.title("📊 ReBloom Analytics Dashboard")
    
    # Add refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Fetch data
    with st.spinner("Fetching data from all sources..."):
        # Fetch Bubble API data
        endpoints = ['Listing', 'User', 'Match']
        endpoint_data = {}
        
        for endpoint in endpoints:
            data = fetch_all_pages(endpoint)
            if data:
                endpoint_data[endpoint] = data
        
        # Fetch Google Analytics data
        website_traffic = get_website_traffic()
        country_traffic = get_country_traffic()
        search_console_data = get_search_console_data()
        active_users_data = get_active_users()
        user_types_data = get_user_types()
        country_active_users = get_country_active_users()
    
    # Display metrics in tabs
    tab1, tab2, tab3 = st.tabs(["Platform Metrics", "Website Analytics", "Search Performance"])
    
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
    
    # Add timestamp
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 