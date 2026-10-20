'use client';
import { useEffect, useRef } from 'react';
import styles from './FLPipeline.module.css';

/**
 * FLPipelineCanvas
 * Animated SVG canvas showing the federated learning flow.
 * Five stages: Courtroom Nodes → Privacy Barrier → Aggregation Hub → CI Gate → Broadcast
 */

const STAGES = [
  {
    id: 1,
    icon: '🏛️',
    title: 'Courtroom Nodes',
    subtitle: 'Local Whisper + LoRA',
    detail: 'Tamil speech inference\n& clerk-in-the-loop SFT',
    accent: '#62B8E8',
  },
  {
    id: 2,
    icon: '🛡️',
    title: 'Privacy Barrier',
    subtitle: 'L₂ Norm Clipping',
    detail: 'ΔW extracted, audio\n& transcripts destroyed',
    accent: '#9B5EF5',
  },
  {
    id: 3,
    icon: '⚡',
    title: 'Aggregation Hub',
    subtitle: 'Secure FedAvg',
    detail: 'ΔW_global = Σ wᵢ · ΔWᵢ\nre-clipped server-side',
    accent: '#D4AE54',
  },
  {
    id: 4,
    icon: '🎯',
    title: 'CI Quality Gate',
    subtitle: 'WER Regression Test',
    detail: 'Held-out Tamil eval\nzero-regression check',
    accent: '#10B981',
  },
  {
    id: 5,
    icon: '🌐',
    title: 'Global Broadcast',
    subtitle: 'Promoted LoRA Adapter',
    detail: 'New adapter version\ndistributed to all courts',
    accent: '#3B82F6',
  },
];

export default function FLPipelineCanvas({ activeStage = 1, metrics = null, adapterVersion = 0 }) {
  const particleRefs = useRef({});

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.headerTitle}>Live Federated Learning Flow</span>
        <span className={styles.headerSub}>
          Stage {activeStage} / 5 active
          {metrics && <span className={styles.metricsTag}>{metrics.participants} courts · {metrics.total_samples} samples</span>}
        </span>
      </div>

      {/* Main pipeline row */}
      <div className={styles.pipelineRow}>
        {STAGES.map((stage, idx) => {
          const isActive = stage.id === activeStage;
          const isPast = stage.id < activeStage;
          return (
            <div key={stage.id} className={styles.stageWrapper}>
              <div
                className={`${styles.node} ${isActive ? styles.nodeActive : isPast ? styles.nodePast : styles.nodeIdle}`}
                style={isActive ? { '--accent': stage.accent, '--glow': stage.accent + '44' } : {}}
              >
                <div className={styles.nodeIcon}>{stage.icon}</div>
                <div className={styles.nodeTitle}>{stage.title}</div>
                <div className={styles.nodeSub}>{stage.subtitle}</div>
                <div className={styles.nodeDetail}>{stage.detail}</div>
                {isActive && <div className={styles.activePulse} style={{ background: stage.accent }} />}
                {isPast && <div className={styles.checkMark}>✓</div>}
              </div>

              {idx < STAGES.length - 1 && (
                <div className={styles.connectorWrapper}>
                  <Connector
                    active={stage.id === activeStage || stage.id === activeStage - 1}
                    accent={stage.accent}
                    reverse={false}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Return flow */}
      <div className={styles.returnFlow}>
        <div className={styles.returnArrows}>
          {[...Array(5)].map((_, i) => (
            <span
              key={i}
              className={styles.returnArrow}
              style={{ animationDelay: `${i * 0.18}s`, opacity: activeStage === 5 ? 1 : 0.3 }}
            >←</span>
          ))}
        </div>
        <div className={styles.returnBadge} style={{ opacity: activeStage === 5 ? 1 : 0.4 }}>
          🌐 Global LoRA Adapter <span className={styles.versionTag}>v{adapterVersion}</span> distributed to all court jurisdictions
        </div>
        <div className={styles.returnArrows}>
          {[...Array(5)].map((_, i) => (
            <span
              key={i}
              className={styles.returnArrow}
              style={{ animationDelay: `${i * 0.18}s`, opacity: activeStage === 5 ? 1 : 0.3 }}
            >←</span>
          ))}
        </div>
      </div>

      {/* Metrics row (shown after FL round) */}
      {metrics && (
        <div className={styles.metricsRow}>
          <MetricPill label="Participants" value={metrics.participants} />
          <MetricPill label="Total Samples" value={metrics.total_samples} />
          <MetricPill label="Local Samples" value={metrics.local_samples} />
          <MetricPill label="Norm Before Clip" value={metrics.local_norm_before_clip} mono />
          <MetricPill label="Aggregate Norm" value={metrics.global_aggregate_norm} mono />
        </div>
      )}
    </div>
  );
}

function Connector({ active, accent, reverse }) {
  return (
    <div className={`${styles.connector} ${active ? styles.connectorActive : ''}`}
      style={active ? { '--connector-color': accent } : {}}>
      <div className={styles.connectorLine} />
      {active && (
        <>
          <div className={styles.particle} style={{ '--delay': '0s', '--color': accent }} />
          <div className={styles.particle} style={{ '--delay': '0.4s', '--color': accent }} />
          <div className={styles.particle} style={{ '--delay': '0.8s', '--color': accent }} />
        </>
      )}
      <div className={styles.connectorArrow} style={active ? { color: accent } : {}}>▶</div>
    </div>
  );
}

function MetricPill({ label, value, mono }) {
  return (
    <div className={styles.metricPill}>
      <span className={styles.metricLabel}>{label}</span>
      <span className={`${styles.metricValue} ${mono ? styles.metricMono : ''}`}>{value}</span>
    </div>
  );
}
