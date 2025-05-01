from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_google_analytics_service():
    """Initialize and return Google Analytics service"""
    try:
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/analytics.readonly']
        )
        service = build('analyticsdata', 'v1beta', credentials=credentials)
        return service
    except Exception as e:
        print(f"Error initializing Google Analytics service: {str(e)}")
        return None

def get_search_console_service():
    """Initialize and return Google Search Console service"""
    try:
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/webmasters.readonly']
        )
        service = build('searchconsole', 'v1', credentials=credentials)
        return service
    except Exception as e:
        print(f"Error initializing Search Console service: {str(e)}")
        return None

def get_website_traffic(days=14):
    """Get website traffic data from Google Analytics"""
    try:
        service = get_google_analytics_service()
        if not service:
            return None

        property_id = os.getenv('GA_VIEW_ID')
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        response = service.properties().runReport(
            property=f"properties/{property_id}",
            body={
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "dimensions": [{"name": "date"}],
                "metrics": [
                    {"name": "sessions"},
                    {"name": "totalUsers"},
                    {"name": "screenPageViews"},
                    {"name": "averageSessionDuration"}
                ]
            }
        ).execute()

        data = []
        for row in response.get('rows', []):
            date = row['dimensionValues'][0]['value']
            metrics = row['metricValues']
            data.append({
                'date': date,
                'sessions': int(metrics[0]['value']),
                'users': int(metrics[1]['value']),
                'pageviews': int(metrics[2]['value']),
                'avg_session_duration': float(metrics[3]['value'])
            })

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching website traffic: {str(e)}")
        return None

def get_country_traffic(days=14):
    """Get traffic by country from Google Analytics"""
    try:
        service = get_google_analytics_service()
        if not service:
            return None

        property_id = os.getenv('GA_VIEW_ID')
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        response = service.properties().runReport(
            property=f"properties/{property_id}",
            body={
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "dimensions": [{"name": "country"}],
                "metrics": [
                    {"name": "sessions"},
                    {"name": "totalUsers"}
                ]
            }
        ).execute()

        data = []
        for row in response.get('rows', []):
            country = row['dimensionValues'][0]['value']
            metrics = row['metricValues']
            data.append({
                'country': country,
                'sessions': int(metrics[0]['value']),
                'users': int(metrics[1]['value'])
            })

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching country traffic: {str(e)}")
        return None

def get_search_console_data(days=28):
    """Get search performance data from Google Search Console"""
    try:
        service = get_search_console_service()
        if not service:
            return None

        site_url = "https://rebloom.ai"
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        request = {
            'startDate': start_date,
            'endDate': end_date,
            'dimensions': ['query', 'page', 'country', 'device'],
            'rowLimit': 1000,
            'startRow': 0
        }

        response = service.searchanalytics().query(
            siteUrl=site_url,
            body=request
        ).execute()

        data = []
        if 'rows' in response:
            for row in response['rows']:
                data.append({
                    'query': row['keys'][0],
                    'page': row['keys'][1],
                    'country': row['keys'][2],
                    'device': row['keys'][3],
                    'clicks': row['clicks'],
                    'impressions': row['impressions'],
                    'ctr': row['ctr'],
                    'position': row['position']
                })

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching Search Console data: {str(e)}")
        return None

def get_active_users(days=14):
    """Get active users data from Google Analytics"""
    try:
        service = get_google_analytics_service()
        if not service:
            return None

        property_id = os.getenv('GA_VIEW_ID')
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        response = service.properties().runReport(
            property=f"properties/{property_id}",
            body={
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "dimensions": [{"name": "date"}],
                "metrics": [{"name": "activeUsers"}]
            }
        ).execute()

        data = []
        for row in response.get('rows', []):
            date = row['dimensionValues'][0]['value']
            active_users = int(row['metricValues'][0]['value'])
            data.append({'date': date, 'active_users': active_users})

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching active users: {str(e)}")
        return None

def get_user_types(days=14):
    """Get new and returning users data from Google Analytics"""
    try:
        service = get_google_analytics_service()
        if not service:
            return None

        property_id = os.getenv('GA_VIEW_ID')
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        response = service.properties().runReport(
            property=f"properties/{property_id}",
            body={
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "dimensions": [{"name": "date"}, {"name": "newVsReturning"}],
                "metrics": [{"name": "activeUsers"}]
            }
        ).execute()

        data = []
        for row in response.get('rows', []):
            date = row['dimensionValues'][0]['value']
            user_type = row['dimensionValues'][1]['value']
            active_users = int(row['metricValues'][0]['value'])
            data.append({'date': date, 'user_type': user_type, 'active_users': active_users})

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching user types: {str(e)}")
        return None

def get_country_active_users(days=14):
    """Get active users by country from Google Analytics"""
    try:
        service = get_google_analytics_service()
        if not service:
            return None

        property_id = os.getenv('GA_VIEW_ID')
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        response = service.properties().runReport(
            property=f"properties/{property_id}",
            body={
                "dateRanges": [{"startDate": start_date, "endDate": end_date}],
                "dimensions": [{"name": "country"}],
                "metrics": [{"name": "activeUsers"}]
            }
        ).execute()

        data = []
        for row in response.get('rows', []):
            country = row['dimensionValues'][0]['value']
            active_users = int(row['metricValues'][0]['value'])
            data.append({'country': country, 'active_users': active_users})

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching active users by country: {str(e)}")
        return None 