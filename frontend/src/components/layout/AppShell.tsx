import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ChevronDown,
  CircleUserRound,
  Network,
  Search,
  Settings,
  ShieldAlert,
  Users,
} from "lucide-react";

interface AppShellProps {
  children: ReactNode;
}

const navigation = [
  {
    label: "Overview",
    path: "/overview",
    icon: Activity,
  },
  {
    label: "Transactions",
    path: "/transactions",
    icon: ShieldAlert,
  },
  {
    label: "Customers",
    path: "/customers",
    icon: Users,
  },
  {
    label: "Fraud Rings",
    path: "/fraud-rings",
    icon: AlertTriangle,
  },
  {
    label: "Graph",
    path: "/graph",
    icon: Network,
  },
  {
    label: "Models",
    path: "/models",
    icon: BarChart3,
  },
];

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldAlert size={18} strokeWidth={2.2} />
          </div>

          <div className="brand-copy">
            <span className="brand-name">Fraud Intelligence</span>
            <span className="brand-subtitle">Risk Operations</span>
          </div>
        </div>

        <div className="workspace-selector">
          <div className="workspace-avatar">FI</div>

          <div className="workspace-copy">
            <span>Fraud Operations</span>
            <small>Production</small>
          </div>

          <ChevronDown size={15} />
        </div>

        <nav className="sidebar-nav" aria-label="Main navigation">
          <div className="nav-section-label">Workspace</div>

          {navigation.map(({ label, path, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `nav-item${isActive ? " active" : ""}`
              }
            >
              <Icon size={17} strokeWidth={1.9} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `nav-item${isActive ? " active" : ""}`
            }
          >
            <Settings size={17} strokeWidth={1.9} />
            <span>Settings</span>
          </NavLink>

          <div className="sidebar-user">
            <div className="user-avatar">
              <CircleUserRound size={18} />
            </div>

            <div className="user-copy">
              <span>Analyst</span>
              <small>Fraud Operations</small>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-search">
            <Search size={17} />
            <span>Search investigations...</span>
            <kbd>⌘ K</kbd>
          </div>

          <div className="system-status">
            <span className="status-dot" />
            <span>API Connected</span>
          </div>
        </header>

        <div className="page-content">{children}</div>
      </main>
    </div>
  );
}