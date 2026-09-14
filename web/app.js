const API_BASE = "";

const LOW_EXAMPLE = {
  Transaction_Amount: 50,
  Average_Spend: 95,
  Previous_Transactions: 20,
  Account_Age_Days: 800,
  Is_International: "0",
  Merchant_Category: "Food",
  Payment_Method: "Debit Card",
  Device_Type: "Mobile",
  Location: "Mumbai",
  localDate: "2023-06-15T14:30",
};

const HIGH_EXAMPLE = {
  Transaction_Amount: 250,
  Average_Spend: 40,
  Previous_Transactions: 1,
  Account_Age_Days: 15,
  Is_International: "1",
  Merchant_Category: "Travel",
  Payment_Method: "Credit Card",
  Device_Type: "POS",
  Location: "Delhi",
  localDate: "2023-01-02T02:15",
};

const form = document.getElementById("predict-form");
const dateInput = document.getElementById("Transaction_Date_Input");
const statusMsg = document.getElementById("status-msg");
const resultEmpty = document.getElementById("result-empty");
const resultBody = document.getElementById("result-body");
const submitBtn = document.getElementById("submit-btn");

let operatingThreshold = null;

function pad(value) {
  return String(value).padStart(2, "0");
}

function toApiDate(localValue) {
  if (!localValue) {
    throw new Error("Invalid transaction date");
  }
  const [datePart, timePart] = localValue.split("T");
  const [year, month, day] = datePart.split("-");
  const [hour, minute] = timePart.split(":");
  return `${pad(day)}-${pad(month)}-${year} ${pad(hour)}:${pad(minute)}`;
}

function formatPercent(value, digits = 2) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function showError(message) {
  statusMsg.hidden = false;
  statusMsg.textContent = message;
  resultEmpty.hidden = true;
  resultBody.hidden = true;
}

function clearError() {
  statusMsg.hidden = true;
  statusMsg.textContent = "";
}

function fillExample(example) {
  document.getElementById("Transaction_Amount").value = example.Transaction_Amount;
  document.getElementById("Average_Spend").value = example.Average_Spend;
  document.getElementById("Previous_Transactions").value = example.Previous_Transactions;
  document.getElementById("Account_Age_Days").value = example.Account_Age_Days;
  document.getElementById("Is_International").value = example.Is_International;
  document.getElementById("Merchant_Category").value = example.Merchant_Category;
  document.getElementById("Payment_Method").value = example.Payment_Method;
  document.getElementById("Device_Type").value = example.Device_Type;
  document.getElementById("Location").value = example.Location;
  dateInput.value = example.localDate;
}

function payloadFromForm() {
  return {
    Transaction_Amount: Number(document.getElementById("Transaction_Amount").value),
    Average_Spend: Number(document.getElementById("Average_Spend").value),
    Previous_Transactions: Number(document.getElementById("Previous_Transactions").value),
    Account_Age_Days: Number(document.getElementById("Account_Age_Days").value),
    Is_International: Number(document.getElementById("Is_International").value),
    Merchant_Category: document.getElementById("Merchant_Category").value,
    Payment_Method: document.getElementById("Payment_Method").value,
    Device_Type: document.getElementById("Device_Type").value,
    Location: document.getElementById("Location").value,
    Transaction_Date: toApiDate(dateInput.value),
  };
}

function renderResult(data) {
  const threshold = Number(data.threshold);
  operatingThreshold = threshold;
  const flagged = Boolean(data.flagged_for_review);
  const chip = document.getElementById("decision-chip");
  chip.textContent = flagged ? "FLAG FOR REVIEW" : "NOT FLAGGED";
  chip.className = `decision ${flagged ? "flag" : "clear"}`;
  document.getElementById("out-probability").textContent = formatPercent(
    data.predicted_probability
  );
  document.getElementById("out-score").textContent =
    `${Number(data.risk_score).toFixed(2)} / 100`;
  document.getElementById("out-band").textContent = String(data.risk_band).toUpperCase();
  document.getElementById("out-label").textContent = flagged
    ? "Flag for review"
    : "Legitimate";
  document.getElementById("out-threshold").textContent = formatPercent(threshold, 0);
  document.getElementById("threshold-display").textContent = formatPercent(threshold, 0);
  document.getElementById("result-caption").textContent =
    "Scores come from the frozen production pipeline. They are not proof of fraud.";
  resultEmpty.hidden = true;
  resultBody.hidden = false;
}

async function loadHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Unable to load model information");
  }
  operatingThreshold = Number(data.threshold);
  document.getElementById("model-name").textContent = data.model;
  if (data.calibration) {
    document.getElementById("calibration-name").textContent = data.calibration;
  }
  document.getElementById("threshold-display").textContent = formatPercent(
    operatingThreshold,
    0
  );
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  submitBtn.disabled = true;
  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm()),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Unable to score this transaction");
    }
    renderResult(data);
  } catch (error) {
    showError(error.message || "Unable to score this transaction");
  } finally {
    submitBtn.disabled = false;
  }
});

document.getElementById("fill-low").addEventListener("click", () => fillExample(LOW_EXAMPLE));
document.getElementById("fill-high").addEventListener("click", () => fillExample(HIGH_EXAMPLE));

fillExample(LOW_EXAMPLE);
loadHealth().catch((error) => {
  showError(error.message || "Unable to load model information");
});
