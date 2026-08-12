"""Supabase-backed auth and per-user saved-analysis storage for Ostrivo.

Uses the publishable (anon) key only. Row Level Security policies on the
`saved_analyses` table (see supabase_schema.sql) enforce that each user can
only read/write their own rows - the secret/service_role key is never used
here, so this module can never bypass that isolation even if misused.

Kept separate from ostrivo_core.py because it does real network I/O (not
pure/unit-testable the same way); kept separate from app.py to keep that
file from growing even larger.
"""

import pandas as pd
from supabase import create_client

from ostrivo_core import json_safe


def get_supabase_client(url, anon_key):
    """Create a Supabase client. Returns None if not configured."""
    if not url or not anon_key:
        return None
    return create_client(url, anon_key)


def sign_up(client, email, password, industry=None):
    """Create a new account. `industry` (if given) is stored in the user's metadata -
    no separate profiles table needed."""
    payload = {"email": email, "password": password}
    if industry:
        payload["options"] = {"data": {"industry": industry}}
    return client.auth.sign_up(payload)


def sign_in(client, email, password):
    return client.auth.sign_in_with_password({"email": email, "password": password})


def update_industry(client, industry):
    """Update the logged-in user's industry preference in their account metadata."""
    return client.auth.update_user({"data": {"industry": industry}})


def sign_out(client):
    client.auth.sign_out()


def restore_session(client, access_token, refresh_token):
    """Restore a logged-in session from previously stored tokens (e.g. a cookie)."""
    return client.auth.set_session(access_token, refresh_token)


def save_analysis(client, user_id, name, df, clean_report, quality_scores,
                   anomaly_summary, ai_summary, col_labels, original_filename):
    """Save a cleaned dataset plus its computed results so it can be reloaded later."""
    payload = {
        'cleaned_data': df.to_json(orient='split', date_format='iso'),
        'clean_report': json_safe(clean_report),
        'quality_scores': json_safe(quality_scores),
        'anomaly_summary': json_safe(anomaly_summary),
        'ai_summary': ai_summary,
        'col_labels': col_labels,
        'original_filename': original_filename,
    }
    return client.table('saved_analyses').insert({
        'user_id': user_id,
        'name': name,
        'row_count': len(df),
        'column_count': len(df.columns),
        'payload': payload,
    }).execute()


def list_saved_analyses(client, user_id):
    """Return metadata only (not the full payload) for all of a user's saved analyses,
    most recent first."""
    response = (
        client.table('saved_analyses')
        .select('id, name, created_at, row_count, column_count')
        .eq('user_id', user_id)
        .order('created_at', desc=True)
        .execute()
    )
    return response.data


def load_analysis(client, analysis_id):
    """Load one saved analysis's full payload, reconstructing the DataFrame."""
    response = client.table('saved_analyses').select('*').eq('id', analysis_id).single().execute()
    row = response.data
    payload = row['payload']
    df = pd.read_json(payload['cleaned_data'], orient='split')
    return {
        'name': row['name'],
        'created_at': row['created_at'],
        'df': df,
        'clean_report': payload['clean_report'],
        'quality_scores': payload['quality_scores'],
        'anomaly_summary': payload['anomaly_summary'],
        'ai_summary': payload.get('ai_summary'),
        'col_labels': payload.get('col_labels') or {},
        'original_filename': payload.get('original_filename'),
    }


def delete_analysis(client, analysis_id):
    return client.table('saved_analyses').delete().eq('id', analysis_id).execute()
