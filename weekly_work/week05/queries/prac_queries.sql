SELECT 
bdrms,
COUNT(bdrms) as frequency,
COUNT(bdrms)/ COUNT(*) as proportion
FROM housing
GROUP BY bdrms
ORDER BY bdrms DESC;

SELECT MAX(age)
FROM housing
WHERE bdrms = 3;

SELECT AVG(age), bdrms
from housing
GROUP BY bdrms;