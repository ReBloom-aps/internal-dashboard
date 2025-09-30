import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# csv_data = """Listing ⁠Name,Ticketsize,Modified Date
# Monzo,$50-100m,May 26 2025 11:27 pm
# Bolt,$20-50m,May 30 2025 11:10 am
# CiaoCacao,<$1m,May 30 2025 2:25 pm
# CiaoCacao,$5-10m,Jun 10 2025 11:22 am
# Anthropic,$20-50m,Jun 2 2025 5:57 pm"""

def parse_ticket_size(ticket_size_str):
    """Parse ticket size string and return max value in millions"""
    if pd.isna(ticket_size_str) or ticket_size_str == '':
        return 0
    
    ticket_size_str = ticket_size_str.strip()
    
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

def parse_ticket_size_min(ticket_size_str):
    """Parse ticket size string and return minimum value in millions"""
    if pd.isna(ticket_size_str) or ticket_size_str == '':
        return 0
    
    ticket_size_str = ticket_size_str.strip()
    
    if '<$1m' in ticket_size_str:
        return 0.5  # Minimum for "<$1m" is 0
    elif '$1-5m' in ticket_size_str:
        return 1  # Minimum for "$1-5m" is 1
    elif '$5-10m' in ticket_size_str:
        return 5  # Minimum for "$5-10m" is 5
    elif '$10-20m' in ticket_size_str:
        return 10  # Minimum for "$10-20m" is 10
    elif '$20-50m' in ticket_size_str:
        return 20  # Minimum for "$20-50m" is 20
    elif '$50-100m' in ticket_size_str:
        return 50  # Minimum for "$50-100m" is 50
    elif '$100m+' in ticket_size_str:
        return 100  # Minimum for "$100m+" is 100
    elif 'Not specified' in ticket_size_str:
        return 0
    elif 'All' in ticket_size_str:
        return 0.5  # Minimum for "All" could be 0
    else:
        return 0

def calculate_ticket_size_metrics(response_data, modified_date='Modified Date', ticket_size='Ticketsize', listing_name=None):
    """Calculate ticket size metrics for display on dashboard"""
    if not response_data or 'response' not in response_data or 'results' not in response_data['response']:
        return 0, 0, 0, 0, 0  # total_min, total_max, count, average_min, average_max
        
    results = response_data['response']['results']
    if not results:
        return 0, 0, 0, 0, 0
        
    # Parse ticket sizes with error handling (both min and max)
    valid_ticket_sizes_max = []
    valid_ticket_sizes_min = []
    for item in results:
        try:
            size_str = item.get(ticket_size, '')
            if size_str:
                parsed_max = parse_ticket_size(size_str)
                parsed_min = parse_ticket_size_min(size_str)
                if parsed_max > 0:  # Only count non-zero ticket sizes
                    valid_ticket_sizes_max.append(parsed_max)
                    valid_ticket_sizes_min.append(parsed_min)
        except (ValueError, TypeError):
            continue
    
    if not valid_ticket_sizes_max:
        return 0, 0, 0, 0, 0
    
    total_min = sum(valid_ticket_sizes_min)
    total_max = sum(valid_ticket_sizes_max)
    count = len(valid_ticket_sizes_max)
    average_min = total_min / count if count > 0 else 0
    average_max = total_max / count if count > 0 else 0
    
    return total_min, total_max, count, average_min, average_max

def calculate_listing_ticket_metrics(response_data):
    """Calculate metrics for listing ticket sizes (Deal specification data)"""
    return calculate_ticket_size_metrics(
        response_data, 
        modified_date='Modified Date', 
        ticket_size='ticket_size', 
        listing_name='Listing Name'
    )

def calculate_demand_ticket_metrics(response_data):
    """Calculate metrics for demand ticket sizes (Investor preference data)"""
    # For demand side, we only have max values (highest_ticket_size), so min = max
    total_min, total_max, count, avg_min, avg_max = calculate_ticket_size_metrics(
        response_data, 
        modified_date='Modified Date', 
        ticket_size='highest_ticket_size'
    )
    # Return the max values as both min and max since we only have single values
    return total_max, count, avg_max

def calculate_combined_demand_ticket_metrics(investor_preference_data, demand_listing_data):
    """Calculate combined metrics for demand ticket sizes from both investor preferences and demand listings"""
    total_min_combined = 0
    total_max_combined = 0
    count_combined = 0
    
    # Process investor preference data
    if investor_preference_data and 'response' in investor_preference_data and 'results' in investor_preference_data['response']:
        for item in investor_preference_data['response']['results']:
            try:
                size_str = item.get('highest_ticket_size', '')
                min_str = item.get('lowest_ticket_size_value', '')
                if size_str:
                    parsed_max = parse_ticket_size(size_str)
                    if parsed_max > 0:
                        total_max_combined += parsed_max
                        total_min_combined += min_str
                        count_combined += 1
            except (ValueError, TypeError):
                continue
    
    # Process demand listing data
    if demand_listing_data and 'response' in demand_listing_data and 'results' in demand_listing_data['response']:
        for item in demand_listing_data['response']['results']:
            try:
                size_str = item.get('Ticketsize', '')
                if size_str:
                    parsed_max = parse_ticket_size(size_str)
                    parsed_min = parse_ticket_size_min(size_str)
                    if parsed_max > 0:
                        total_max_combined += parsed_max
                        total_min_combined += parsed_min
                        count_combined += 1
            except (ValueError, TypeError):
                continue
    
    return total_min_combined, total_max_combined, count_combined

def create_interactive_plot_ticket_no_listing_name(response_data, title='Accumulated Ticket Size of Platform Listings Over Time', modified_date='Modified Date', ticket_size='Ticketsize'):
    """Create interactive ticket size tracker plots using Plotly without listing names"""
    
    if not response_data or 'response' not in response_data or 'results' not in response_data['response']:
        raise ValueError("Invalid response data format")
        
    results = response_data['response']['results']
    if not results:
        raise ValueError("No results found in response data")
        
    total_count = len(results)
    main_title = f"{title} (Total: {total_count})"
    
    # Parse modified dates with error handling
    modified_dates = []
    for item in results:
        try:
            date_str = item.get(modified_date, '')
            if not date_str:
                raise ValueError(f"Missing {modified_date} field")
            # Handle both ISO format and date-only format
            if 'T' in date_str:
                date_str = date_str[:10]  # Take just the date part from ISO format
            modified_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse date for item: {e}")
            continue
    
    if not modified_dates:
        raise ValueError("No valid dates found in the data")
        
    df = pd.DataFrame({'Modified Date': modified_dates})
    
    # Parse ticket sizes with error handling
    ticket_sizes = []
    for item in results:
        try:
            size_str = item.get(ticket_size, '')
            if not size_str:
                raise ValueError(f"Missing {ticket_size} field")
            ticket_sizes.append(parse_ticket_size(size_str))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse ticket size for item: {e}")
            continue
            
    df['Ticket Size Value'] = ticket_sizes
    
    # Sort by modified date
    df = df.sort_values('Modified Date').reset_index(drop=True)
    
    # Calculate cumulative ticket size
    df['Cumulative Ticket Size'] = df['Ticket Size Value'].cumsum()
    
    # Create subplots with secondary y-axis for the top chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[main_title, f'Daily Ticket Sizes Over Time ({total_count} Entries)'],
        vertical_spacing=0.25,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Main cumulative chart (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['Modified Date'],
            y=df['Cumulative Ticket Size'],
            mode='lines+markers',
            name='Cumulative Total',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB'),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.3)',
            hovertemplate='<b>Cumulative Total</b><br>' +
                         'Cumulative Size: $%{y:.1f}M<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Add individual ticket size markers (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['Modified Date'],
            y=df['Ticket Size Value'],
            mode='markers',
            name='Individual Tickets',
            marker=dict(size=10, color='orange', line=dict(color='black', width=1)),
            hovertemplate='<b>Date: %{x}</b><br>' +
                         'Ticket Size: $%{y:.1f}M<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )
    
    # Create timeline bar chart (bottom subplot) with proper stacking
    # Group entries by date to create stacked bars
    date_groups = df.groupby('Modified Date')
    
    # Create color palette
    colors = px.colors.qualitative.Set3
    
    # Get all unique dates for consistent x-axis
    all_dates = sorted(df['Modified Date'].unique())
    
    # Create stacked bars for each date
    for date_idx, date in enumerate(all_dates):
        date_entries = date_groups.get_group(date)
        
        if len(date_entries) == 1:
            # Single entry for this date
            entry = date_entries.iloc[0]
            hover_text = (
                f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                f"Ticket Size: ${entry['Ticket Size Value']:.1f}M"
            )
            
            fig.add_trace(
                go.Bar(
                    x=[date],
                    y=[entry['Ticket Size Value']],
                    name=f"{date.strftime('%b %d')}",
                    marker_color=colors[date_idx % len(colors)],
                    opacity=0.8,
                    width=86400000 * 0.8,  # Set bar width (80% of day in milliseconds)
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )
        else:
            # Multiple entries for this date - create stacked bars
            cumulative_base = 0
            total_for_date = date_entries['Ticket Size Value'].sum()
            date_entries_list = list(date_entries.iterrows())
            
            for entry_idx, (_, entry) in enumerate(date_entries_list):
                # Only show total for the last (top) entry in the stack
                if entry_idx == len(date_entries_list) - 1:
                    hover_text = (
                        f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                        f"<b>Total for date: ${total_for_date:.1f}M</b><br>" +
                        f"Entry {entry_idx + 1}: ${entry['Ticket Size Value']:.1f}M"
                    )
                else:
                    hover_text = (
                        f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                        f"Entry {entry_idx + 1}: ${entry['Ticket Size Value']:.1f}M"
                    )
                
                fig.add_trace(
                    go.Bar(
                        x=[date],
                        y=[entry['Ticket Size Value']],
                        name=f"{date.strftime('%b %d')} - Entry {entry_idx + 1}",
                        marker_color=colors[(date_idx + entry_idx) % len(colors)],
                        opacity=0.8,
                        width=86400000 * 0.8,  # Set bar width (80% of day in milliseconds)
                        base=[cumulative_base],  # Stack on top of previous bars
                        hovertemplate=hover_text + '<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                cumulative_base += entry['Ticket Size Value']
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Update x-axis for first subplot
    fig.update_xaxes(title_text="Date", row=1, col=1)
    
    # Update y-axes for first subplot (dual y-axis)
    fig.update_yaxes(title_text="Cumulative Ticket Size ($ Millions)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Individual Ticket Size ($ Millions)", row=1, col=1, secondary_y=True)
    
    # Update x-axis for second subplot
    fig.update_xaxes(title_text="Date", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Daily Ticket Size Total ($ Millions)", row=2, col=1)
    
    # Print summary statistics
    print("\n=== TICKET SIZE SUMMARY (No Listing Names) ===")
    print(f"Total Accumulated Ticket Size: ${df['Cumulative Ticket Size'].iloc[-1]:.1f} Million")
    print(f"Number of Entries: {len(df)}")
    print(f"Average Ticket Size: ${df['Ticket Size Value'].mean():.1f} Million")
    print(f"Largest Single Ticket: ${df['Ticket Size Value'].max():.1f}M")
    print(f"Date Range: {df['Modified Date'].min().strftime('%B %d, %Y')} to {df['Modified Date'].max().strftime('%B %d, %Y')}")
    
    print("\n=== ENTRY DETAILS ===")
    for _, row in df.iterrows():
        print(f"• ${row['Ticket Size Value']:.1f}M - {row['Modified Date'].strftime('%b %d, %Y')}")
    
    return fig

def create_interactive_plot_ticket(response_data, title='Accumulated Ticket Size of Platform Listings Over Time', modified_date='Modified Date', ticket_size='Ticketsize', listing_name='Listing ⁠Name', second_response_data=None, filter_func=None):
    """Create interactive ticket size tracker plots using Plotly"""
    
    if not response_data or 'response' not in response_data or 'results' not in response_data['response']:
        raise ValueError("Invalid response data format")
    
    results = response_data['response']['results']
    if not results:
        raise ValueError("No results found in response data")
    print("results before filtering:")
    print(len(results))
    if filter_func:
        results = [item for item in results if filter_func(item)]
    if second_response_data:
        second_results = second_response_data['response']['results']
        
    total_count = len(results)
    main_title = f"{title} (Total: {total_count})"
    
    # Parse modified dates with error handling
    modified_dates = []
    for item in results:
        try:
            date_str = item.get(modified_date, '')
            if not date_str:
                raise ValueError(f"Missing {modified_date} field")
            # Handle both ISO format and date-only format
            if 'T' in date_str:
                date_str = date_str[:10]  # Take just the date part from ISO format
            modified_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse date for item: {e}")
            continue
    
    if not modified_dates:
        raise ValueError("No valid dates found in the data")
        
    df = pd.DataFrame({'Modified Date': modified_dates})
    
    # Parse ticket sizes with error handling (both max and min)
    ticket_sizes_max = []
    ticket_sizes_min = []
    # If secondary data is provided, use its ticket_size by joining on _id (results) and 'OG listing' (second_results)
    og_listing_to_ticket_size = {}
    if second_response_data:
        try:
            og_listing_to_ticket_size = {
                sr.get('OG listing'): sr.get('ticket_size', '')
                for sr in second_results if sr.get('OG listing')
            }
        except Exception as e:
            print(f"Warning: Failed to build secondary ticket size map: {e}")
            og_listing_to_ticket_size = {}

    for item in results:
        try:
            if second_response_data:
                result_id = item.get('_id')
                if not result_id:
                    raise ValueError("Missing _id field for join with secondary results")
                size_str = og_listing_to_ticket_size.get(result_id, '')
                if not size_str:
                    raise ValueError("No matching ticket_size in secondary results for this _id")
            else:
                size_str = item.get(ticket_size, '')
                if not size_str:
                    raise ValueError(f"Missing {ticket_size} field")

            ticket_sizes_max.append(parse_ticket_size(size_str))
            ticket_sizes_min.append(parse_ticket_size_min(size_str))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse ticket size for item: {e}")
            continue
            
    df['Ticket Size Value'] = ticket_sizes_max
    df['Ticket Size Min'] = ticket_sizes_min

    # Add listing names with error handling
    listing_names = []
    for item in results:
        try:
            name = item.get(listing_name, '')
            if not name:
                name = 'Not specified'
                print(f"Warning: Missing Listing Name, using 'Not specified'")
            listing_names.append(name)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not get listing name for item: {e}")
            continue
            
    df['Listing ⁠Name'] = listing_names
    
    # Sort by modified date
    df = df.sort_values('Modified Date').reset_index(drop=True)
    
    # Calculate cumulative ticket size (max and min)
    df['Cumulative Ticket Size'] = df['Ticket Size Value'].cumsum()
    df['Cumulative Ticket Size Min'] = df['Ticket Size Min'].cumsum()
    
    # Prepare data for counting unique listings
    unique_listings = sorted(df['Listing ⁠Name'].unique())  # Sort alphabetically
    listing_count = len(unique_listings)
    
    # Create subplots with secondary y-axis for the top chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[main_title, f'Supply Ticket Sizes ({listing_count} companies)'],
        vertical_spacing=0.25,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Main cumulative chart (primary y-axis) with shaded range
    # Add the minimum cumulative line (light blue) for fill reference
    fig.add_trace(
        go.Scatter(
            x=df['Modified Date'],
            y=df['Cumulative Ticket Size Min'],
            mode='lines',
            name='Cumulative Min',
            line=dict(color='#2ED9FF', width=2),  # Light blue line
            showlegend=True,
            hovertemplate='<b>Cumulative Minimum</b><br>' +
                         'Date: %{x}<br>' +
                         'Min: $%{y:.1f}M<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Add the maximum cumulative line with fill to minimum
    fig.add_trace(
        go.Scatter(
            x=df['Modified Date'],
            y=df['Cumulative Ticket Size'],
            mode='lines+markers',
            name='Cumulative Max',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB'),
            fill='tonexty',  # Fill to previous trace (the min line)
            fillcolor='rgba(46, 134, 171, 0.3)',
            hovertemplate='<b>Cumulative Range</b><br>' +
                         'Date: %{x}<br>' +
                         'Max: $%{y:.1f}M<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Add individual listing markers with hover info (secondary y-axis)
    for i, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['Modified Date']],
                y=[row['Ticket Size Value']],
                mode='markers',
                name=f"{row['Listing ⁠Name']}",
                marker=dict(size=12, color='yellow', line=dict(color='black', width=1)),
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Range: $%{text:.1f}M - $%{y:.1f}M<extra></extra>',
                text=[row['Ticket Size Min']]
            ),
            row=1, col=1, secondary_y=True
        )
    
    # Prepare data for stacked bar chart
    listing_groups = df.groupby('Listing ⁠Name')
    
    # Create color palette
    colors = px.colors.qualitative.Set3
    
    # Create stacked bar chart with proper stacking and thick bars
    for i, listing in enumerate(unique_listings):
        group_data = listing_groups.get_group(listing)
        
        # Convert to list to allow proper indexing
        group_rows = list(group_data.iterrows())
        
        if len(group_rows) > 1:
            # If multiple entries for same listing, stack them properly
            cumulative_bottom = 0
            total_for_listing = group_data['Ticket Size Value'].sum()
            for j, (_, row) in enumerate(group_rows):
                # Only show total for the last (top) entry in the stack
                if j == len(group_rows) - 1:
                    hover_text = (
                        f"<b>{row['Listing ⁠Name']}</b><br>" +
                        f"<b>Total for listing: ${total_for_listing:.1f}M</b><br>" +
                        f"Entry {j + 1}: {row['Modified Date'].strftime('%b %d, %Y')}<br>" +
                        f"Range: ${row['Ticket Size Min']:.1f}M - ${row['Ticket Size Value']:.1f}M"
                    )
                else:
                    hover_text = (
                        f"<b>{row['Listing ⁠Name']}</b><br>" +
                        f"Entry {j + 1}: {row['Modified Date'].strftime('%b %d, %Y')}<br>" +
                        f"Range: ${row['Ticket Size Min']:.1f}M - ${row['Ticket Size Value']:.1f}M"
                    )
                
                fig.add_trace(
                    go.Bar(
                        x=[listing],
                        y=[row['Ticket Size Value']],
                        name=f"{listing} - {row['Modified Date'].strftime('%b %d, %Y')}",
                        marker_color=colors[(i + j) % len(colors)],
                        opacity=0.8,
                        width=0.8,  # Set bar width (80% of category width)
                        base=[cumulative_bottom],  # Stack on top of previous bars
                        hovertemplate=hover_text + '<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                cumulative_bottom += row['Ticket Size Value']
        else:
            # Single entry for listing
            row = group_rows[0][1]  # Get the row data
            hover_text = (
                f"<b>{row['Listing ⁠Name']}</b><br>" +
                f"Date: {row['Modified Date'].strftime('%b %d, %Y')}<br>" +
                f"Range: ${row['Ticket Size Min']:.1f}M - ${row['Ticket Size Value']:.1f}M"
            )
            
            fig.add_trace(
                go.Bar(
                    x=[listing],
                    y=[row['Ticket Size Value']],
                    name=listing,
                    marker_color=colors[i % len(colors)],
                    opacity=0.8,
                    width=0.8,  # Set bar width (80% of category width)
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Value labels removed for cleaner visualization
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        # title_text="Interactive Ticket Size Analysis",
        # title_x=0.5,
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Update x-axis for first subplot
    fig.update_xaxes(title_text="Updated Date", row=1, col=1)
    
    # Update y-axes for first subplot (dual y-axis)
    fig.update_yaxes(title_text="Cumulative Ticket Size ($ Millions)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Individual Ticket Size ($ Millions)", row=1, col=1, secondary_y=True)
    
    # Update x-axis for second subplot
    fig.update_xaxes(title_text="Listings", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Ticket Size ($ Millions)", row=2, col=1)
    
    # Print summary statistics
    print("\n=== PLATFORM TICKET SIZE SUMMARY ===")
    print(f"Total Accumulated Ticket Size: ${df['Cumulative Ticket Size'].iloc[-1]:.1f} Million")
    print(f"Number of Listings: {len(df)}")
    print(f"Average Ticket Size: ${df['Ticket Size Value'].mean():.1f} Million")
    print(f"Largest Single Ticket: {df.loc[df['Ticket Size Value'].idxmax(), 'Listing ⁠Name']} (${df['Ticket Size Value'].max():.1f}M)")
    print(f"Date Range: {df['Modified Date'].min().strftime('%B %d, %Y')} to {df['Modified Date'].max().strftime('%B %d, %Y')}")
    
    print("\n=== LISTING DETAILS ===")
    for _, row in df.iterrows():
        print(f"• {row['Listing ⁠Name']}: ${row['Ticket Size Value']:.1f}M - {row['Modified Date'].strftime('%b %d, %Y')}")
    
    return fig

# To use with your own CSV file, replace the csv_data with:
def load_from_csv_file(file_path):
    """Load data from actual CSV file"""
    df = pd.read_csv(file_path)
    
    # Parse modified dates (adjust format as needed)
    df['Modified Date'] = pd.to_datetime(df['Modified Date'])
    
    # Parse ticket sizes
    df['Ticket Size Value'] = df['Ticketsize'].apply(parse_ticket_size)
    
    # Sort by modified date
    df = df.sort_values('Modified Date').reset_index(drop=True)
    
    # Calculate cumulative ticket size
    df['Cumulative Ticket Size'] = df['Ticket Size Value'].cumsum()
    
    return df

def fetch_all_pages(endpoint_name):
    """Fetch all pages of data from an endpoint"""
    base_url = f"https://rebloom.ai/version-test/api/1.1/obj/{endpoint_name}"
    # headers = {
    #     'Content-Type': 'application/json',
    #     'Authorization': f'Bearer {os.getenv("BUBBLE_API_TOKEN")}'
    # }
    all_results = []
    cursor = 0
    
    while True:
        # Add fields parameter to explicitly request fields
        url = f"{base_url}?cursor={cursor}"
        if endpoint_name == 'User':
            # Request all relevant user fields
            url += "&fields=Created Date,Modified Date,_id,email,user_signed_up,user_last_login,user_name,user_role,user_status,user_type,user_verified,user_phone,user_address,user_city,user_state,user_zip,user_country,user_company,user_website,user_linkedin,user_twitter,user_facebook,user_instagram,user_youtube,user_github,user_avatar,user_bio,user_tags,user_preferences,user_settings,user_notifications,user_metadata"
        
        response = requests.get(base_url)
        
        if response.status_code == 200:
            data = response.json()
            results = data['response']['results']
            all_results.extend(results)
            
            remaining = data['response'].get('remaining', 0)
            if remaining == 0:
                break
                
            cursor += len(results)
        else:
            print(f"Error fetching page: {response.status_code}")
            print(f"Response: {response.text}")
            break
    
    return {'response': {'results': all_results, 'count': len(all_results), 'remaining': 0}}

# Run the analysis
# if __name__ == "__main__":
#     # For demo with sample data
#     response_data = fetch_all_pages('Demand_listing')
#     print(response_data)
#     result_df = create_ticket_size_tracker(response_data)
    
    # To use with your CSV file, uncomment and modify:
    # df = load_from_csv_file('your_file.csv')
    # Then create similar visualizations


# def create_ticket_size_tracker(response_data, title = 'Accumulated Ticket Size of Platform Listings Over Time', modified_date = 'Modified Date', ticket_size = 'Ticketsize', listing_name = 'Listing ⁠Name'):
#     if not response_data or 'response' not in response_data or 'results' not in response_data['response']:
#         raise ValueError("Invalid response data format")
        
#     results = response_data['response']['results']
#     if not results:
#         raise ValueError("No results found in response data")
        
#     total_count = len(results)
#     title = f"{title} (Total: {total_count})"
    
#     # Parse modified dates with error handling
#     modified_dates = []
#     for item in results:
#         try:
#             date_str = item.get(modified_date, '')
#             if not date_str:
#                 raise ValueError(f"Missing {modified_date} field")
#             # Handle both ISO format and date-only format
#             if 'T' in date_str:
#                 date_str = date_str[:10]  # Take just the date part from ISO format
#             modified_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
#         except (ValueError, TypeError) as e:
#             print(f"Warning: Could not parse date for item: {e}")
#             continue
    
#     if not modified_dates:
#         raise ValueError("No valid dates found in the data")
        
#     df = pd.DataFrame({'Modified Date': modified_dates})
    
#     # Parse ticket sizes with error handling
#     ticket_sizes = []
#     for item in results:
#         try:
#             size_str = item.get(ticket_size, '')
#             if not size_str:
#                 raise ValueError(f"Missing {ticket_size} field")
#             ticket_sizes.append(parse_ticket_size(size_str))
#         except (ValueError, TypeError) as e:
#             print(f"Warning: Could not parse ticket size for item: {e}")
#             continue
            
#     df['Ticket Size Value'] = ticket_sizes

#     # Add listing names with error handling
#     listing_names = []
#     for item in results:
#         try:
#             name = item.get(listing_name, '')
#             if not name:
#                 raise ValueError(f"Missing {listing_name} field")
#             listing_names.append(name)
#         except (ValueError, TypeError) as e:
#             print(f"Warning: Could not get listing name for item: {e}")
#             continue
            
#     df['Listing ⁠Name'] = listing_names
    
#     # Sort by modified date
#     df = df.sort_values('Modified Date').reset_index(drop=True)
    
#     # Calculate cumulative ticket size
#     df['Cumulative Ticket Size'] = df['Ticket Size Value'].cumsum()
    
#     # Create the visualization
#     plt.style.use('seaborn-v0_8')
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
#     # Main cumulative chart
#     ax1.plot(df['Modified Date'], df['Cumulative Ticket Size'], 
#              marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
#     ax1.fill_between(df['Modified Date'], df['Cumulative Ticket Size'], 
#                      alpha=0.3, color='#2E86AB')
    
#     # Add individual listing markers with labels
#     for i, row in df.iterrows():
#         ax1.annotate(f"{row['Listing ⁠Name']}\n(${row['Ticket Size Value']:.1f}M)", 
#                     xy=(row['Modified Date'], row['Cumulative Ticket Size']),
#                     xytext=(10, 10), textcoords='offset points',
#                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
#                     fontsize=9, ha='left')
    
#     ax1.set_title(title, 
#                   fontsize=16, fontweight='bold', pad=20)
#     ax1.set_xlabel('Modified Date', fontsize=12)
#     ax1.set_ylabel('Cumulative Ticket Size ($ Millions)', fontsize=12)
#     ax1.grid(True, alpha=0.3)
#     ax1.tick_params(axis='x', rotation=45)
    
#     # Prepare data for stacked bar chart
#     # Group by listing name and create stacked data
#     listing_groups = df.groupby('Listing ⁠Name')
#     unique_listings = df['Listing ⁠Name'].unique()
    
#     # Create stacked bar chart
#     bottom = np.zeros(len(unique_listings))
#     # Use a colormap with enough colors for all possible entries
#     colors = plt.cm.Set3(np.linspace(0, 1, max(len(df), 12)))  # Ensure at least 12 colors
#     color_idx = 0
    
#     # Plot each group's bars
#     for listing in unique_listings:
#         group_data = listing_groups.get_group(listing)
#         for _, row in group_data.iterrows():
#             ax2.bar(listing, row['Ticket Size Value'], bottom=bottom[list(unique_listings).index(listing)],
#                    color=colors[color_idx % len(colors)], alpha=0.8, label=f"{row['Modified Date'].strftime('%b %d, %Y')}")
#             bottom[list(unique_listings).index(listing)] += row['Ticket Size Value']
#             color_idx += 1
    
#     # Add value labels on bars
#     for i, listing in enumerate(unique_listings):
#         total_height = bottom[i]
#         ax2.text(i, total_height + 0.5, f'${total_height:.1f}M', 
#                 ha='center', va='bottom', fontweight='bold')
    
#     ax2.set_title('Individual Listing Ticket Sizes (Stacked)', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('Listings', fontsize=12)
#     ax2.set_ylabel('Ticket Size ($ Millions)', fontsize=12)
#     ax2.tick_params(axis='x', rotation=45)
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     # Add legend for stacked segments
#     handles, labels = ax2.get_legend_handles_labels()
#     ax2.legend(handles, labels, title='Modified Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
    
#     # Print summary statistics
#     print("\n=== PLATFORM TICKET SIZE SUMMARY ===")
#     print(f"Total Accumulated Ticket Size: ${df['Cumulative Ticket Size'].iloc[-1]:.1f} Million")
#     print(f"Number of Listings: {len(df)}")
#     print(f"Average Ticket Size: ${df['Ticket Size Value'].mean():.1f} Million")
#     print(f"Largest Single Ticket: {df.loc[df['Ticket Size Value'].idxmax(), 'Listing ⁠Name']} (${df['Ticket Size Value'].max():.1f}M)")
#     print(f"Date Range: {df['Modified Date'].min().strftime('%B %d, %Y')} to {df['Modified Date'].max().strftime('%B %d, %Y')}")
    
#     print("\n=== LISTING DETAILS ===")
#     for _, row in df.iterrows():
#         print(f"• {row['Listing ⁠Name']}: ${row['Ticket Size Value']:.1f}M - {row['Modified Date'].strftime('%b %d, %Y')}")
    
#     plt.show()
    
#     return df

def create_combined_demand_plot(investor_preference_data, demand_listing_data, title='Demand Side Ticket Size', modified_date='Modified Date'):
    """Create interactive combined plot for investor preferences and demand listings with different colors"""
    
    # Validate data
    if not investor_preference_data or 'response' not in investor_preference_data or 'results' not in investor_preference_data['response']:
        raise ValueError("Invalid investor preference data format")
    if not demand_listing_data or 'response' not in demand_listing_data or 'results' not in demand_listing_data['response']:
        raise ValueError("Invalid demand listing data format")
        
    investor_results = investor_preference_data['response']['results']
    demand_results = demand_listing_data['response']['results']
    
    if not investor_results and not demand_results:
        raise ValueError("No results found in either dataset")
        
    # Process investor preference data
    investor_dates = []
    investor_ticket_sizes_max = []
    investor_ticket_sizes_min = []
    
    for item in investor_results:
        try:
            date_str = item.get(modified_date, '')
            if not date_str:
                continue
            # Handle both ISO format and date-only format
            if 'T' in date_str:
                date_str = date_str[:10]
            investor_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            
            size_str = item.get('highest_ticket_size', '')
            investor_ticket_sizes_max.append(parse_ticket_size(size_str))
            investor_ticket_sizes_min.append(parse_ticket_size_min(size_str))
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse investor preference item: {e}")
            continue
    
    # Process demand listing data
    demand_dates = []
    demand_ticket_sizes_max = []
    demand_ticket_sizes_min = []
    demand_listing_names = []
    
    for item in demand_results:
        try:
            date_str = item.get(modified_date, '')
            if not date_str:
                continue
            # Handle both ISO format and date-only format
            if 'T' in date_str:
                date_str = date_str[:10]
            demand_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            
            size_str = item.get('Ticketsize', '')
            demand_ticket_sizes_max.append(parse_ticket_size(size_str))
            demand_ticket_sizes_min.append(parse_ticket_size_min(size_str))
            
            listing_name = item.get('Listing ⁠Name', 'Not specified')
            demand_listing_names.append(listing_name)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse demand listing item: {e}")
            continue
    
    # Create combined DataFrame
    combined_data = []
    
    # Add investor preference data
    for date, size_max, size_min in zip(investor_dates, investor_ticket_sizes_max, investor_ticket_sizes_min):
        combined_data.append({
            'Date': date,
            'Ticket Size': size_max,
            'Ticket Size Min': size_min,
            'Source': 'Investor Preference',
            'Listing Name': None
        })
    
    # Add demand listing data
    for date, size_max, size_min, listing in zip(demand_dates, demand_ticket_sizes_max, demand_ticket_sizes_min, demand_listing_names):
        combined_data.append({
            'Date': date,
            'Ticket Size': size_max,
            'Ticket Size Min': size_min,
            'Source': 'Demand Listing',
            'Listing Name': listing
        })
    
    if not combined_data:
        raise ValueError("No valid data found in either dataset")
    
    df = pd.DataFrame(combined_data)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate cumulative ticket size (max and min)
    df['Cumulative Ticket Size'] = df['Ticket Size'].cumsum()
    df['Cumulative Ticket Size Min'] = df['Ticket Size Min'].cumsum()
    
    total_count = len(df)
    investor_count = len([x for x in combined_data if x['Source'] == 'Investor Preference'])
    demand_count = len([x for x in combined_data if x['Source'] == 'Demand Listing'])
    
    main_title = f"{title} (Total: {total_count} | Public Demand: {investor_count} | Darkpool Demand: {demand_count})"
    
    # Create subplots with secondary y-axis for the top chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[main_title, f'Daily Demand Ticket Sizes ({total_count} Entries)'],
        vertical_spacing=0.25,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Main cumulative chart (primary y-axis) with shaded range
    # Add the minimum cumulative line (light blue) for fill reference
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Cumulative Ticket Size Min'],
            mode='lines',
            name='Cumulative Min',
            line=dict(color='#2ED9FF', width=2),  # Light blue line
            showlegend=True,
            hovertemplate='<b>Cumulative Minimum</b><br>' +
                         'Date: %{x}<br>' +
                         'Min: $%{y:.1f}M<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Add the maximum cumulative line with fill to minimum
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Cumulative Ticket Size'],
            mode='lines+markers',
            name='Cumulative Max',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB'),
            fill='tonexty',  # Fill to previous trace (the min line)
            fillcolor='rgba(46, 134, 171, 0.3)',
            hovertemplate='<b>Cumulative Range</b><br>' +
                         'Date: %{x}<br>' +
                         'Max: $%{y:.1f}M<extra></extra>'
        ),
        row=1, col=1, secondary_y=False
    )
    
    # Add individual markers with different colors for each source (secondary y-axis)
    investor_df = df[df['Source'] == 'Investor Preference']
    demand_df = df[df['Source'] == 'Demand Listing']
    
    # Investor preference markers (single orange color)
    if not investor_df.empty:
        fig.add_trace(
            go.Scatter(
                x=investor_df['Date'],
                y=investor_df['Ticket Size'],
                mode='markers',
                name='Public(Investor Preferences)',
                marker=dict(size=10, color='orange', line=dict(color='black', width=1)),
                hovertemplate='<b>Investor Preference</b><br>' +
                             'Date: %{x}<br>' +
                             'Ticket Size: $%{y:.1f}M<extra></extra>',
                showlegend=True
            ),
            row=1, col=1, secondary_y=True
        )
    
    # Demand listing markers (yellow) with listing names
    if not demand_df.empty:
        hover_text = []
        for _, row in demand_df.iterrows():
            hover_text.append(
                f"<b>Demand Listing</b><br>" +
                f"Listing: {row['Listing Name']}<br>" +
                f"Date: {row['Date'].strftime('%Y-%m-%d')}<br>" +
                f"Ticket Size: ${row['Ticket Size']:.1f}M"
            )
        
        fig.add_trace(
            go.Scatter(
                x=demand_df['Date'],
                y=demand_df['Ticket Size'],
                mode='markers',
                name='Darkpool(Demand Listings)',
                marker=dict(size=10, color='yellow', line=dict(color='black', width=1)),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1, secondary_y=True
        )
    
    # Create timeline bar chart (bottom subplot) with proper stacking by date
    date_groups = df.groupby('Date')
    colors = px.colors.qualitative.Set3
    
    # Get all unique dates for consistent x-axis
    all_dates = sorted(df['Date'].unique())
    
    # Create stacked bars for each date with color coding by source
    orange_colors = ['#FF8C00', '#FF7F50', '#FF6347', '#FF4500', '#FFA500', '#FFB347', '#FFCC99']
    investor_color_idx = 0
    
    for date_idx, date in enumerate(all_dates):
        date_entries = date_groups.get_group(date)
        
        if len(date_entries) == 1:
            # Single entry for this date
            entry = date_entries.iloc[0]
            if entry['Source'] == 'Investor Preference':
                color = orange_colors[investor_color_idx % len(orange_colors)]
                investor_color_idx += 1
            else:
                color = 'yellow'
            
            if entry['Source'] == 'Demand Listing':
                hover_text = (
                    f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                    f"Source: {entry['Source']}<br>" +
                    f"Listing: {entry['Listing Name']}<br>" +
                    f"Ticket Size: ${entry['Ticket Size']:.1f}M"
                )
            else:
                hover_text = (
                    f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                    f"Source: {entry['Source']}<br>" +
                    f"Ticket Size: ${entry['Ticket Size']:.1f}M"
                )
            
            fig.add_trace(
                go.Bar(
                    x=[date],
                    y=[entry['Ticket Size']],
                    name=f"{date.strftime('%b %d')} - {entry['Source'][:8]}",
                    marker_color=color,
                    opacity=0.8,
                    width=86400000 * 0.8,
                    hovertemplate=hover_text + '<extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )
        else:
            # Multiple entries for this date - create stacked bars
            cumulative_base = 0
            total_for_date = date_entries['Ticket Size'].sum()
            date_entries_list = list(date_entries.iterrows())
            
            for entry_idx, (_, entry) in enumerate(date_entries_list):
                if entry['Source'] == 'Investor Preference':
                    color = orange_colors[investor_color_idx % len(orange_colors)]
                    investor_color_idx += 1
                else:
                    color = 'yellow'
                
                # Only show total for the last (top) entry in the stack
                if entry_idx == len(date_entries_list) - 1:
                    if entry['Source'] == 'Demand Listing':
                        hover_text = (
                            f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                            f"<b>Total for date: ${total_for_date:.1f}M</b><br>" +
                            f"Entry {entry_idx + 1}: {entry['Source']}<br>" +
                            f"Listing: {entry['Listing Name']}<br>" +
                            f"Ticket Size: ${entry['Ticket Size']:.1f}M"
                        )
                    else:
                        hover_text = (
                            f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                            f"<b>Total for date: ${total_for_date:.1f}M</b><br>" +
                            f"Entry {entry_idx + 1}: {entry['Source']}<br>" +
                            f"Ticket Size: ${entry['Ticket Size']:.1f}M"
                        )
                else:
                    if entry['Source'] == 'Demand Listing':
                        hover_text = (
                            f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                            f"Entry {entry_idx + 1}: {entry['Source']}<br>" +
                            f"Listing: {entry['Listing Name']}<br>" +
                            f"Ticket Size: ${entry['Ticket Size']:.1f}M"
                        )
                    else:
                        hover_text = (
                            f"<b>Date: {date.strftime('%b %d, %Y')}</b><br>" +
                            f"Entry {entry_idx + 1}: {entry['Source']}<br>" +
                            f"Ticket Size: ${entry['Ticket Size']:.1f}M"
                        )
                
                fig.add_trace(
                    go.Bar(
                        x=[date],
                        y=[entry['Ticket Size']],
                        name=f"{date.strftime('%b %d')} - {entry['Source'][:8]} {entry_idx + 1}",
                        marker_color=color,
                        opacity=0.8,
                        width=86400000 * 0.8,
                        base=[cumulative_base],
                        hovertemplate=hover_text + '<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                cumulative_base += entry['Ticket Size']
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Update x-axis for first subplot
    fig.update_xaxes(title_text="Date", row=1, col=1)
    
    # Update y-axes for first subplot (dual y-axis)
    fig.update_yaxes(title_text="Cumulative Ticket Size ($ Millions)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Individual Ticket Size ($ Millions)", row=1, col=1, secondary_y=True)
    
    # Update x-axis for second subplot
    fig.update_xaxes(title_text="Date", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="Daily Ticket Size Total ($ Millions)", row=2, col=1)
    
    # Print summary statistics
    print("\n=== COMBINED DEMAND TICKET SIZE SUMMARY ===")
    print(f"Total Accumulated Ticket Size: ${df['Cumulative Ticket Size'].iloc[-1]:.1f} Million")
    print(f"Total Entries: {len(df)}")
    print(f"  - Investor Preferences: {investor_count}")
    print(f"  - Demand Listings: {demand_count}")
    print(f"Average Ticket Size: ${df['Ticket Size'].mean():.1f} Million")
    print(f"Largest Single Ticket: ${df['Ticket Size'].max():.1f}M")
    print(f"Date Range: {df['Date'].min().strftime('%B %d, %Y')} to {df['Date'].max().strftime('%B %d, %Y')}")
    
    print("\n=== ENTRY DETAILS ===")
    for _, row in df.iterrows():
        if row['Source'] == 'Demand Listing':
            print(f"• {row['Source']}: {row['Listing Name']} - ${row['Ticket Size']:.1f}M - {row['Date'].strftime('%b %d, %Y')}")
        else:
            print(f"• {row['Source']}: ${row['Ticket Size']:.1f}M - {row['Date'].strftime('%b %d, %Y')}")
    
    return fig
