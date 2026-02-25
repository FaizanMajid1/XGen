-- Run this in your Supabase project: SQL Editor → New query → paste and run
-- Creates the table for persistent, shared history across all devices

CREATE TABLE IF NOT EXISTS tweetgen_history (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  module TEXT NOT NULL,
  inputs JSONB NOT NULL DEFAULT '{}',
  result JSONB NOT NULL DEFAULT '{}'
);

-- Allow the app to read and write (when using service_role key, RLS is bypassed)
-- If using anon key, enable these policies:
-- ALTER TABLE tweetgen_history ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Allow all for tweetgen_history" ON tweetgen_history FOR ALL USING (true) WITH CHECK (true);
