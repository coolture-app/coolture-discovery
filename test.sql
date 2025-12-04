CREATE TABLE events (
  id BIGSERIAL PRIMARY KEY,
  title TEXT,
  content TEXT,
  category TEXT,
  author TEXT,
  created_at TIMESTAMP DEFAULT now(),
  embedding VECTOR(1536)
);

SELECT id, title, content, 1 - (embedding <=> '') AS similarity
FROM events
ORDER BY similarity
LIMIT 5;