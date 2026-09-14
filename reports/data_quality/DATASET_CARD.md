# Fraud Intelligence Synthetic Dataset Card

## 1. Dataset Overview

**Dataset name:** fraud-intelligence-synthetic  
**Dataset version:** 1.0.0  
**Dataset type:** Fully synthetic relational financial transaction dataset  
**Purpose:** Fraud detection, graph machine learning, explainable AI, and fraud-ring investigation  
**Generation seed:** 42  
**Primary development size:** 100,000 transactions  
**Target fraud rate:** 1%

This dataset was generated specifically for the Fraud Intelligence project.

It does not contain real customer, financial, device, IP, or merchant information.

---

## 2. Business Objective

The dataset is designed to support a fraud intelligence system capable of answering:

1. Is this transaction suspicious?
2. Why is it suspicious?
3. What entities are connected to the transaction?
4. Is the transaction part of a larger fraud ring?

The dataset intentionally contains relational structures that can later be represented as a heterogeneous graph.

---

## 3. Entities

The synthetic environment contains the following entities:

| Entity | Configured Count |
|---|---:|
| Customers | 20,000 |
| Accounts | 25,000 |
| Cards | 30,000 |
| Devices | 25,000 |
| IPs | 30,000 |
| Merchants | 1,000 |
| Transactions | 100,000 |

---

## 4. Transaction Schema

The transaction dataset contains 14 columns:

| Column | Description |
|---|---|
| transaction_id | Unique transaction identifier |
| timestamp | Transaction timestamp in UTC |
| customer_id | Customer associated with transaction |
| account_id | Account used for transaction |
| card_id | Card associated with account |
| merchant_id | Merchant involved in transaction |
| device_id | Device associated with transaction |
| ip_id | IP address identifier |
| amount | Synthetic transaction amount |
| currency | Synthetic transaction currency |
| payment_method | Payment mechanism |
| transaction_type | Type of transaction |
| is_fraud | Binary fraud label |
| fraud_scenario | Fraud scenario responsible for injected fraud |

---

## 5. Fraud Scenarios

The dataset contains seven synthetic fraud scenarios:

| Scenario | Configured Weight |
|---|---:|
| ACCOUNT_TAKEOVER | 20% |
| SHARED_DEVICE | 15% |
| SHARED_IP | 15% |
| MERCHANT_ABUSE | 15% |
| VELOCITY_ATTACK | 15% |
| GEO_ANOMALY | 5% |
| FRAUD_RING | 15% |

The configured fraud rate is 1%.

For the frozen 100,000-transaction dataset, exactly 1,000 transactions are labeled as fraudulent.

Observed fraud counts:

| Scenario | Transactions |
|---|---:|
| ACCOUNT_TAKEOVER | 188 |
| SHARED_IP | 168 |
| FRAUD_RING | 157 |
| SHARED_DEVICE | 149 |
| VELOCITY_ATTACK | 147 |
| MERCHANT_ABUSE | 138 |
| GEO_ANOMALY | 53 |
| **Total** | **1,000** |

---

## 6. Fraud Injection Design

### Account Takeover

Fraudulent transactions are associated with unusual devices/IPs and elevated transaction amounts.

### Shared Device

Multiple suspicious transactions are associated with a device that is not normally associated with the customer's activity.

### Shared IP

Fraudulent activity is associated with suspicious shared IP infrastructure.

### Merchant Abuse

Suspicious activity is concentrated around selected merchants with elevated transaction amounts.

### Velocity Attack

Multiple suspicious transactions are clustered in time for the same customer.

### Geo Anomaly

Transactions are associated with unusual device/IP combinations representing anomalous geographic behavior.

### Fraud Ring

Multiple customers share infrastructure such as:

- device
- IP
- merchant

Fraud-ring transactions are temporally clustered and associated with elevated amounts.

The fraud-ring design intentionally creates graph structures that can later be detected using graph analytics and graph machine learning.

---

## 7. Temporal Design

Transaction timestamps are generated between:

**2025-01-01 00:00:00 UTC**

and

**2025-12-31 00:00:00 UTC**

The upper boundary is exclusive.

Observed frozen dataset range:

- Minimum: 2025-01-01 00:00:03 UTC
- Maximum: 2025-12-30 23:49:24 UTC

All timestamps are timezone-aware UTC timestamps.

---

## 8. Relational Integrity

The dataset maintains the following relationships:

```text
Customer
   ├── Account
   │      └── Card
   │
   ├── Device
   ├── IP
   └── Merchant preferences

Transaction
   ├── Customer
   ├── Account
   ├── Card
   ├── Device
   ├── IP
   └── Merchant