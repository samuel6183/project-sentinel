-- Run against data/transactions.sqlite3 after generating the dataset.
SELECT provider, COUNT(*) AS attempts,
 AVG(CASE WHEN status='approved' THEN 1.0 ELSE 0.0 END) AS approval_rate,
 AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END) AS failure_rate,
 SUM(amount) AS attempted_volume_kes,
 SUM(ABS(actual_settlement-expected_settlement)) AS absolute_discrepancy_kes
FROM transactions GROUP BY provider ORDER BY provider;

SELECT substr(timestamp,1,10) AS day, COUNT(*) AS attempts
FROM transactions GROUP BY day ORDER BY day;

SELECT transaction_id,provider,expected_settlement,actual_settlement,
 ROUND(actual_settlement-expected_settlement,2) AS difference_kes
FROM transactions WHERE ROUND(actual_settlement-expected_settlement,2)<>0;
