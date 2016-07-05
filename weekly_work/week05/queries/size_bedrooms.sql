SELECT 
bdrms,
COUNT(bdrms) as frequency
FROM housing
GROUP BY bdrms
ORDER BY bdrms DESC;