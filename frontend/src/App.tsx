import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import "./styles/app.css";

import { AppShell } from "./components/layout/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { TransactionsPage } from "./pages/TransactionsPage";
import { CustomersPage } from "./pages/CustomersPage";
import FraudRingsPage from "./pages/FraudRingsPage";
import GraphPage from "./pages/GraphPage";
import { ModelsPage } from "./pages/ModelsPage";
import SettingsPage from "./pages/SettingsPage";

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<DashboardPage />} />
          <Route path="/transactions" element={<TransactionsPage />} />
          <Route path="/customers" element={<CustomersPage />} />
          <Route path="/fraud-rings" element={<FraudRingsPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;
