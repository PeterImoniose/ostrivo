-- Run this once in the Supabase SQL Editor for your project.
-- Creates the table that stores each user's saved analyses, with Row Level
-- Security so a user can only ever see, insert, update, or delete their own rows.

create extension if not exists pgcrypto;

create table if not exists saved_analyses (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    name text not null,
    created_at timestamptz not null default now(),
    row_count integer,
    column_count integer,
    payload jsonb not null
);

create index if not exists idx_saved_analyses_user_id on saved_analyses(user_id);

alter table saved_analyses enable row level security;

create policy "select_own_analyses" on saved_analyses
    for select using (auth.uid() = user_id);

create policy "insert_own_analyses" on saved_analyses
    for insert with check (auth.uid() = user_id);

create policy "update_own_analyses" on saved_analyses
    for update using (auth.uid() = user_id);

create policy "delete_own_analyses" on saved_analyses
    for delete using (auth.uid() = user_id);
