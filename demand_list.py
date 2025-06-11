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
    """Parse ticket size string and return midpoint value in millions"""
    if pd.isna(ticket_size_str) or ticket_size_str == '':
        return 0
    
    ticket_size_str = ticket_size_str.strip()
    
    if '<$1m' in ticket_size_str:
        return 1  # Assume midpoint of 0-1m
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

def create_ticket_size_tracker(response_data, title = 'Accumulated Ticket Size of Platform Listings Over Time', modified_date = 'Modified Date', ticket_size = 'Ticketsize', listing_name = 'Listing ⁠Name'):
    if not response_data or 'response' not in response_data or 'results' not in response_data['response']:
        raise ValueError("Invalid response data format")
        
    results = response_data['response']['results']
    if not results:
        raise ValueError("No results found in response data")
        
    total_count = len(results)
    title = f"{title} (Total: {total_count})"
    
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

    # Add listing names with error handling
    listing_names = []
    for item in results:
        try:
            name = item.get(listing_name, '')
            if not name:
                raise ValueError(f"Missing {listing_name} field")
            listing_names.append(name)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not get listing name for item: {e}")
            continue
            
    df['Listing ⁠Name'] = listing_names
    
    # Sort by modified date
    df = df.sort_values('Modified Date').reset_index(drop=True)
    
    # Calculate cumulative ticket size
    df['Cumulative Ticket Size'] = df['Ticket Size Value'].cumsum()
    
    # Create the visualization
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Main cumulative chart
    ax1.plot(df['Modified Date'], df['Cumulative Ticket Size'], 
             marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    ax1.fill_between(df['Modified Date'], df['Cumulative Ticket Size'], 
                     alpha=0.3, color='#2E86AB')
    
    # Add individual listing markers with labels
    for i, row in df.iterrows():
        ax1.annotate(f"{row['Listing ⁠Name']}\n(${row['Ticket Size Value']:.1f}M)", 
                    xy=(row['Modified Date'], row['Cumulative Ticket Size']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=9, ha='left')
    
    ax1.set_title(title, 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Modified Date', fontsize=12)
    ax1.set_ylabel('Cumulative Ticket Size ($ Millions)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Prepare data for stacked bar chart
    # Group by listing name and create stacked data
    listing_groups = df.groupby('Listing ⁠Name')
    unique_listings = df['Listing ⁠Name'].unique()
    
    # Create stacked bar chart
    bottom = np.zeros(len(unique_listings))
    # Use a colormap with enough colors for all possible entries
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(df), 12)))  # Ensure at least 12 colors
    color_idx = 0
    
    # Plot each group's bars
    for listing in unique_listings:
        group_data = listing_groups.get_group(listing)
        for _, row in group_data.iterrows():
            ax2.bar(listing, row['Ticket Size Value'], bottom=bottom[list(unique_listings).index(listing)],
                   color=colors[color_idx % len(colors)], alpha=0.8, label=f"{row['Modified Date'].strftime('%b %d, %Y')}")
            bottom[list(unique_listings).index(listing)] += row['Ticket Size Value']
            color_idx += 1
    
    # Add value labels on bars
    for i, listing in enumerate(unique_listings):
        total_height = bottom[i]
        ax2.text(i, total_height + 0.5, f'${total_height:.1f}M', 
                ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Individual Listing Ticket Sizes (Stacked)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Listings', fontsize=12)
    ax2.set_ylabel('Ticket Size ($ Millions)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add legend for stacked segments
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, title='Modified Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
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
    
    plt.show()
    
    return df

def create_interactive_plot_ticket(response_data, title='Accumulated Ticket Size of Platform Listings Over Time', modified_date='Modified Date', ticket_size='Ticketsize', listing_name='Listing ⁠Name'):
    """Create interactive ticket size tracker plots using Plotly"""
    
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

    # Add listing names with error handling
    listing_names = []
    for item in results:
        try:
            name = item.get(listing_name, '')
            if not name:
                raise ValueError(f"Missing {listing_name} field")
            listing_names.append(name)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not get listing name for item: {e}")
            continue
            
    df['Listing ⁠Name'] = listing_names
    
    # Sort by modified date
    df = df.sort_values('Modified Date').reset_index(drop=True)
    
    # Calculate cumulative ticket size
    df['Cumulative Ticket Size'] = df['Ticket Size Value'].cumsum()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[main_title, 'Individual Listing Ticket Sizes (Stacked)'],
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # Main cumulative chart
    fig.add_trace(
        go.Scatter(
            x=df['Modified Date'],
            y=df['Cumulative Ticket Size'],
            mode='lines+markers',
            name='Cumulative Ticket Size',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8, color='#2E86AB'),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.3)',
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Cumulative Size:</b> $%{y:.1f}M<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add individual listing markers with hover info
    for i, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row['Modified Date']],
                y=[row['Cumulative Ticket Size']],
                mode='markers',
                name=f"{row['Listing ⁠Name']}",
                marker=dict(size=12, color='yellow', line=dict(color='black', width=1)),
                showlegend=False,
                hovertemplate='<b>' + row['Listing ⁠Name'] + '</b><br>' +
                             '<b>Individual Ticket Size:</b> $' + f'{row["Ticket Size Value"]:.1f}' + 'M<br>' +
                             '<b>Date:</b> %{x}<br>' +
                             '<b>Cumulative Size:</b> $%{y:.1f}M<br>' +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Prepare data for stacked bar chart
    listing_groups = df.groupby('Listing ⁠Name')
    unique_listings = df['Listing ⁠Name'].unique()
    
    # Create color palette
    colors = px.colors.qualitative.Set3
    
    # Create stacked bar chart
    for i, listing in enumerate(unique_listings):
        group_data = listing_groups.get_group(listing)
        y_values = []
        hover_texts = []
        
        for j, (_, row) in enumerate(group_data.iterrows()):
            y_values.append(row['Ticket Size Value'])
            hover_texts.append(
                f"<b>{row['Listing ⁠Name']}</b><br>" +
                f"Date: {row['Modified Date'].strftime('%b %d, %Y')}<br>" +
                f"Ticket Size: ${row['Ticket Size Value']:.1f}M"
            )
        
        # Calculate bottom position for stacking
        if len(group_data) > 1:
            # If multiple entries for same listing, stack them
            for j, (_, row) in enumerate(group_data.iterrows()):
                bottom = sum([r['Ticket Size Value'] for _, r in group_data.iterrows()[:j]])
                fig.add_trace(
                    go.Bar(
                        x=[listing],
                        y=[row['Ticket Size Value']],
                        name=f"{listing} - {row['Modified Date'].strftime('%b %d, %Y')}",
                        marker_color=colors[j % len(colors)],
                        opacity=0.8,
                        hovertemplate=hover_texts[j] + '<extra></extra>',
                        base=bottom if j > 0 else 0
                    ),
                    row=2, col=1
                )
        else:
            # Single entry for listing
            fig.add_trace(
                go.Bar(
                    x=[listing],
                    y=y_values,
                    name=listing,
                    marker_color=colors[i % len(colors)],
                    opacity=0.8,
                    hovertemplate=hover_texts[0] + '<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Add value labels on bars
    for i, listing in enumerate(unique_listings):
        group_data = listing_groups.get_group(listing)
        total_height = group_data['Ticket Size Value'].sum()
        fig.add_annotation(
            x=listing,
            y=total_height + 0.5,
            text=f'${total_height:.1f}M',
            showarrow=False,
            font=dict(size=12, color='black'),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Interactive Ticket Size Analysis",
        title_x=0.5,
        template="plotly_white"
    )
    
    # Update x-axis for first subplot
    fig.update_xaxes(title_text="Modified Date", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Ticket Size ($ Millions)", row=1, col=1)
    
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