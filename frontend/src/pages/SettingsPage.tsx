import { useEffect, useState } from "react";
import {
  CheckCircle2,
  Database,
  GitBranch,
  Server,
  ShieldCheck,
} from "lucide-react";

import {
  getHealth,
  getSystemMetadata,
} from "../api/client";

import type {
  HealthResponse,
  SystemMetadata,
} from "../types/system";

function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metadata, setMetadata] = useState<SystemMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadSystemInformation() {
      setLoading(true);
      setError("");

      try {
        const [healthResponse, metadataResponse] = await Promise.all([
          getHealth(),
          getSystemMetadata(),
        ]);

        if (!active) {
          return;
        }

        setHealth(healthResponse);
        setMetadata(metadataResponse);
      } catch (requestError) {
        if (!active) {
          return;
        }

        setError(
          requestError instanceof Error
            ? requestError.message
            : "Unable to load system information.",
        );
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadSystemInformation();

    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return (
      <section className="page">
        <div className="page-header">
          <div>
            <span className="eyebrow">Configuration</span>
            <h1>Settings</h1>
            <p>Application and investigation configuration.</p>
          </div>
        </div>

        <div className="settings-loading">
          <Server size={22} />
          <span>Loading system information...</span>
        </div>
      </section>
    );
  }

  if (error || !health || !metadata) {
    return (
      <section className="page">
        <div className="page-header">
          <div>
            <span className="eyebrow">Configuration</span>
            <h1>Settings</h1>
            <p>Application and investigation configuration.</p>
          </div>
        </div>

        <div className="settings-error">
          <Server size={20} />
          <div>
            <strong>Unable to load system information</strong>
            <span>{error || "System metadata is unavailable."}</span>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="page">
      <div className="page-header settings-heading">
        <div>
          <span className="eyebrow">Configuration</span>
          <h1>System Settings</h1>
          <p>
            Read-only application, API, dataset, and architecture information.
          </p>
        </div>

        <div className="system-status">
          <CheckCircle2 size={17} />
          <span>Operational</span>
        </div>
      </div>

      <section className="settings-overview-grid">
        <div className="settings-overview-card">
          <div className="settings-icon">
            <Server size={18} />
          </div>

          <span>API service</span>
          <strong>{health.service}</strong>
          <small>Version {health.version}</small>
        </div>

        <div className="settings-overview-card">
          <div className="settings-icon">
            <Database size={18} />
          </div>

          <span>Dataset</span>
          <strong>{metadata.dataset.name}</strong>
          <small>Version {metadata.dataset.version}</small>
        </div>

        <div className="settings-overview-card">
          <div className="settings-icon">
            <GitBranch size={18} />
          </div>

          <span>Project phase</span>
          <strong>{metadata.phase}</strong>
          <small>API metadata version</small>
        </div>

        <div className="settings-overview-card">
          <div className="settings-icon">
            <ShieldCheck size={18} />
          </div>

          <span>System status</span>
          <strong>{health.status}</strong>
          <small>Health endpoint response</small>
        </div>
      </section>

      <section className="settings-panel">
        <div className="settings-panel-header">
          <div>
            <span className="eyebrow">API CONFIGURATION</span>
            <h2>Runtime information</h2>
          </div>

          <span className="settings-readonly">READ ONLY</span>
        </div>

        <div className="settings-detail-grid">
          <div>
            <span>Service</span>
            <strong>{health.service}</strong>
          </div>

          <div>
            <span>API version</span>
            <strong>{health.version}</strong>
          </div>

          <div>
            <span>Project</span>
            <strong>{metadata.project}</strong>
          </div>

          <div>
            <span>Operational status</span>
            <strong>{metadata.status}</strong>
          </div>

          <div className="settings-full-width">
            <span>Frontend API base URL</span>
            <code>
              {import.meta.env.VITE_API_BASE_URL ||
                "http://127.0.0.1:8000"}
            </code>
          </div>
        </div>
      </section>

      <section className="settings-panel">
        <div className="settings-panel-header">
          <div>
            <span className="eyebrow">ARCHITECTURE</span>
            <h2>Pipeline components</h2>
          </div>
        </div>

        <div className="architecture-list">
          {Object.entries(metadata.architecture).map(
            ([component, phase]) => (
              <div className="architecture-row" key={component}>
                <span>
                  {component.replace(/_/g, " ")}
                </span>

                <strong>{phase}</strong>
              </div>
            ),
          )}
        </div>
      </section>

      <section className="settings-panel">
        <div className="settings-panel-header">
          <div>
            <span className="eyebrow">DATASET</span>
            <h2>Dataset configuration</h2>
          </div>
        </div>

        <div className="dataset-detail">
          <div>
            <span>Dataset name</span>
            <strong>{metadata.dataset.name}</strong>
          </div>

          <div>
            <span>Dataset version</span>
            <strong>{metadata.dataset.version}</strong>
          </div>
        </div>
      </section>

      <section className="settings-panel">
        <div className="settings-panel-header">
          <div>
            <span className="eyebrow">PROJECT GOVERNANCE</span>
            <h2>Frozen phases</h2>
          </div>

          <span className="settings-readonly">
            {metadata.frozen_phases.length} phases
          </span>
        </div>

        <div className="frozen-phase-list">
          {metadata.frozen_phases.map((phase) => (
            <span key={phase}>
              <CheckCircle2 size={14} />
              {phase}
            </span>
          ))}
        </div>
      </section>

      <div className="settings-disclaimer">
        <strong>Read-only configuration</strong>
        <span>
          This interface exposes runtime and project metadata. It does not
          modify model artifacts, datasets, API configuration, or investigation
          logic.
        </span>
      </div>
    </section>
  );
}

export default SettingsPage;