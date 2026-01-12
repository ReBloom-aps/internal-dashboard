import os
import requests
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter

# Set the style
plt.style.use('bmh')  # Using a built-in matplotlib style

# Set font family and other styling parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def print_json_structure(data, endpoint_name):
    """Print a structured overview of the JSON response"""
    print(f"\n=== {endpoint_name} Structure Overview ===")
    
    # Print basic response info
    print(f"\nTotal Results: {data['response'].get('count', 'N/A')}")
    print(f"Remaining: {data['response'].get('remaining', 'N/A')}")
    
    if 'results' in data['response']:
        results = data['response']['results']
        if results:
            # Get sample record
            sample = results[0]
            
            # Print structure of a record
            print("\nRecord Structure:")
            for key in sorted(sample.keys()):
                value = sample[key]
                value_type = type(value).__name__
                # Truncate long values but show more context
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:50] + "..."
                print(f"  {key}: ({value_type}) Example: {value_str}")
            
            # Collect some statistics
            print("\nData Statistics:")
            print(f"Number of records: {len(results)}")
            
            # Count status types if status field exists
            if 'status' in sample:
                status_counts = Counter(item['status'] for item in results)
                print("\nStatus Distribution:")
                for status, count in status_counts.items():
                    print(f"  {status}: {count}")
            
            # Date range if Created Date exists
            if 'Created Date' in sample:
                dates = [datetime.strptime(item['Created Date'][:10], '%Y-%m-%d') 
                        for item in results]
                print(f"\nDate Range:")
                print(f"  Earliest: {min(dates).date()}")
                print(f"  Latest: {max(dates).date()}")
            
            # Additional User-specific statistics
            if endpoint_name == 'User':
                if 'user_role' in sample:
                    role_counts = Counter(item.get('user_role', 'N/A') for item in results)
                    print("\nUser Role Distribution:")
                    for role, count in role_counts.items():
                        print(f"  {role}: {count}")
                
                if 'user_status' in sample:
                    status_counts = Counter(item.get('user_status', 'N/A') for item in results)
                    print("\nUser Status Distribution:")
                    for status, count in status_counts.items():
                        print(f"  {status}: {count}")
            
            print("\n" + "="*50)  # Separator between endpoints

def fetch_all_pages(endpoint_name):
    """Fetch all pages of data from an endpoint"""
    base_url = f"{os.getenv('BUBBLE_ENDPOINT')}/{endpoint_name}"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("BUBBLE_API_TOKEN")}'
    }
    all_results = []
    cursor = 0
    
    while True:
        # Add fields parameter to explicitly request fields
        url = f"{base_url}?cursor={cursor}"
        if endpoint_name == 'User':
            # Request all relevant user fields (including investor_profile)
            url += "&fields=Created Date,Modified Date,_id,email,user_signed_up,user_last_login,user_name,user_role,user_status,user_type,user_verified,user_phone,user_address,user_city,user_state,user_zip,user_country,user_company,user_website,user_linkedin,user_twitter,user_facebook,user_instagram,user_youtube,user_github,user_avatar,user_bio,user_tags,user_preferences,user_settings,user_notifications,user_metadata,investor_profile"
        
        response = requests.get(url, headers=headers)
        
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

def create_growth_plot(data, date_field, title, output_file, filter_func=None):
    """Create a growth plot for the given data"""
    # Filter data if filter function is provided
    results = data['response']['results']
    if filter_func:
        results = [item for item in results if filter_func(item)]
    
    total_count = len(results)
    title = f"{title} (Total: {total_count})"
    
    # Extract creation dates and convert to datetime
    creation_dates = [datetime.strptime(item[date_field][:10], '%Y-%m-%d') 
                     for item in results]
    
    # Create DataFrame with dates
    df = pd.DataFrame({'Created Date': creation_dates})
    
    # Count daily creations
    daily_counts = df.groupby('Created Date').size()
    
    # Calculate cumulative sum
    cumulative_counts = daily_counts.cumsum()
    
    # Calculate week-over-week growth
    latest_date = max(daily_counts.index)
    week_ago_date = latest_date - timedelta(days=7)
    
    # Find the closest available date for week-ago comparison
    available_dates = sorted(daily_counts.index)
    week_ago_date = max([d for d in available_dates if d <= week_ago_date], default=None)
    
    current_total = cumulative_counts.iloc[-1]
    week_ago_total = cumulative_counts.get(week_ago_date, 0) if week_ago_date else 0
    
    if week_ago_total > 0:
        growth_percentage = ((current_total - week_ago_total) / week_ago_total) * 100
        growth_text = f"Week-over-Week Growth: +{growth_percentage:.1f}%\n(+{current_total - week_ago_total} total)"
    else:
        if len(available_dates) > 1:
            days_of_data = (latest_date - min(available_dates)).days
            growth_text = f"Growth data limited\n({days_of_data} days of history available)"
        else:
            growth_text = "Insufficient data for growth calculation"
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8), facecolor='white')
    ax2 = ax1.twinx()
    
    # Plot histogram on first y-axis with gradient color
    bars = ax1.bar(daily_counts.index, daily_counts.values, alpha=0.7, 
                  color='#2ecc71', label='Daily Creations', width=0.8)
    
    # Add subtle shadow to bars
    for bar in bars:
        bar.set_edgecolor('none')
        
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold', labelpad=15)
    ax1.set_ylabel('Daily Creations', color='#2ecc71', 
                  fontsize=12, fontweight='bold', labelpad=15)
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    
    # Plot cumulative line on second y-axis
    line = ax2.plot(cumulative_counts.index, cumulative_counts.values, 
                   color='#9b59b6', linewidth=3, label='Cumulative Total')
    ax2.set_ylabel('Cumulative Total', color='#9b59b6', 
                  fontsize=12, fontweight='bold', labelpad=15)
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    
    # Style improvements
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add title with custom styling
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add growth annotation
    plt.text(0.5, 0.95, growth_text,
            transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
            verticalalignment='top',
            horizontalalignment='center',
            fontsize=12,
            color='#34495e',
            fontweight='bold')
    
    # Adjust layout
    fig.tight_layout()
    
    # Add legends with custom styling
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', frameon=True, 
              fancybox=True, shadow=True, 
              fontsize=10)
    
    # Add a subtle background color
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f8f9fa')
    
    # Adjust margins
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
    # Save the plot with high DPI
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")
    plt.close()

def export_institutional_users_csv(user_data, output_csv_path):
    """Filter users with investor_profile == 'Institutional investor' and export to CSV"""
    if not user_data or 'response' not in user_data or 'results' not in user_data['response']:
        return 0
    results = user_data['response']['results']
    institutional = [
        item for item in results
        if item.get('investor_profile') == 'Institutional investor'
    ]
    if not institutional:
        # Write empty CSV with no rows but predictable columns
        pd.DataFrame(columns=['Created Date', '_id', 'full_name']).to_csv(output_csv_path, index=False)
        return 0
    df = pd.DataFrame(institutional)
    # Keep only the specified columns
    columns_to_keep = ['Created Date', '_id', 'full_name']
    # Only keep columns that exist in the dataframe
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[available_columns]
    df.to_csv(output_csv_path, index=False)
    return len(df)

def export_institutional_mom_2025(user_data, date_field, output_csv_path, output_png_path):
    """Create 2025 month-over-month signup counts for institutional users and export CSV + PNG."""
    if not user_data or 'response' not in user_data or 'results' not in user_data['response']:
        return 0
    results = user_data['response']['results']
    institutional = [
        item for item in results
        if item.get('investor_profile') == 'Institutional investor' and item.get(date_field)
    ]
    if not institutional:
        # Write empty CSV
        pd.DataFrame(columns=['Month', 'Signups', 'Institution Names']).to_csv(output_csv_path, index=False)
        return 0
    # Parse dates and filter for 2025, keeping institution names
    monthly_data = []
    for item in institutional:
        try:
            date_str = item[date_field]
            date_obj = datetime.strptime(date_str[:10], '%Y-%m-%d')
            if date_obj.year == 2025:
                month = date_obj.replace(day=1)  # First day of month
                institution_name = item.get('full_name', 'N/A')
                monthly_data.append({
                    'month': month,
                    'institution_name': institution_name
                })
        except Exception:
            continue
    if not monthly_data:
        pd.DataFrame(columns=['Month', 'Signups', 'Institution Names']).to_csv(output_csv_path, index=False)
        return 0
    df = pd.DataFrame(monthly_data)
    # Group by month and aggregate signups and institution names separately
    monthly_counts = df.groupby('month').size().reset_index(name='Signups')
    monthly_names = df.groupby('month')['institution_name'].apply(lambda x: ', '.join(x)).reset_index(name='Institution Names')
    monthly = monthly_counts.merge(monthly_names, on='month', how='outer')
    monthly.columns = ['Month', 'Signups', 'Institution Names']
    # Reorder columns
    monthly = monthly[['Month', 'Signups', 'Institution Names']]
    # Ensure all months of 2025 are present
    all_months = pd.date_range('2025-01-01', '2025-12-01', freq='MS')
    monthly = monthly.set_index('Month').reindex(all_months, fill_value=0).rename_axis('Month').reset_index()
    # Fill missing values
    monthly['Signups'] = monthly['Signups'].fillna(0).astype(int)
    monthly['Institution Names'] = monthly['Institution Names'].fillna('')
    # Save CSV
    monthly.to_csv(output_csv_path, index=False)
    # Plot PNG with institution names as labels
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    bars = ax.bar(monthly['Month'], monthly['Signups'], color='#2ecc71', alpha=0.7, edgecolor='none')
    ax.set_title('Institutional Signups MoM (2025)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Signups', fontsize=14, fontweight='bold')
    ax.set_facecolor('#f8f9fa')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    # Add institution names as text annotations above bars
    for idx, row in monthly.iterrows():
        if row['Signups'] > 0 and row['Institution Names']:
            institution_names = row['Institution Names']
            # Split long names into multiple lines if needed
            max_chars_per_line = 30
            if len(institution_names) > max_chars_per_line:
                # Split by comma and create multi-line text
                names_list = institution_names.split(', ')
                lines = []
                current_line = ''
                for name in names_list:
                    if len(current_line) + len(name) + 2 <= max_chars_per_line:
                        current_line += (', ' if current_line else '') + name
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = name
                if current_line:
                    lines.append(current_line)
                display_text = '\n'.join(lines[:3])  # Limit to 3 lines max
                if len(lines) > 3:
                    display_text += f'\n... (+{len(lines)-3} more)'
            else:
                display_text = institution_names
            
            # Position text above the bar
            y_pos = row['Signups'] + max(0.5, row['Signups'] * 0.05)
            ax.text(row['Month'], y_pos, display_text,
                   ha='center', va='bottom', fontsize=7, color='#34495e',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#ccc'))
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    return int(monthly['Signups'].sum())

def main():
    # Load environment variables
    load_dotenv()
    
    # Fetch and analyze each endpoint with all pages
    endpoints = ['Listing', 'User', 'Match']
    endpoint_data = {}
    
    for endpoint in endpoints:
        data = fetch_all_pages(endpoint)
        if data:
            print_json_structure(data, endpoint)
            endpoint_data[endpoint] = data
    
    # Create plots for each type
    if 'Listing' in endpoint_data:
        create_growth_plot(
            endpoint_data['Listing'],
            'Created Date',
            'Listing Creation Timeline',
            'listing_creations.png'
        )
    
    if 'User' in endpoint_data:
        create_growth_plot(
            endpoint_data['User'],
            'Created Date',
            'User Growth Timeline',
            'user_growth.png'
        )
        # Exports for institutional users
        os.makedirs('exports', exist_ok=True)
        csv_users = os.path.join('exports', 'institutional_users.csv')
        csv_mom = os.path.join('exports', 'institutional_signups_2025_mom.csv')
        png_mom = os.path.join('exports', 'institutional_signups_2025_mom.png')
        exported_count = export_institutional_users_csv(endpoint_data['User'], csv_users)
        export_institutional_mom_2025(endpoint_data['User'], 'Created Date', csv_mom, png_mom)
    
    if 'Match' in endpoint_data:
        # Filter for active seller matches
        create_growth_plot(
            endpoint_data['Match'],
            'Created Date',
            'Active Match Timeline',
            'active_matches.png',
            filter_func=lambda x: x.get('seller_match_status') == 'Active'
        )

if __name__ == "__main__":
    main()