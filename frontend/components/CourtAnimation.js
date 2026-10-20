'use client';

import React, { useState } from 'react';
import styles from './CourtAnimation.module.css';

/**
 * CourtAnimation Component
 * A rich, prestigious courtroom animation featuring the Scales of Justice,
 * classical columns, acoustic wave ribbons, and a rotating judicial privacy seal.
 */
export default function CourtAnimation() {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className={styles.animationContainer}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      title="Judicial Balance of Justice & Privacy Guarantee"
    >
      <svg
        className={`${styles.courtSvg} ${isHovered ? styles.svgActive : ''}`}
        viewBox="0 0 520 380"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          {/* Gradients */}
          <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#F59E0B" />
            <stop offset="35%" stopColor="#D97706" />
            <stop offset="70%" stopColor="#B45309" />
            <stop offset="100%" stopColor="#78350F" />
          </linearGradient>

          <linearGradient id="brassGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#FDE68A" />
            <stop offset="50%" stopColor="#D97706" />
            <stop offset="100%" stopColor="#92400E" />
          </linearGradient>

          <linearGradient id="pillarGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#CBD5E1" />
            <stop offset="30%" stopColor="#F8FAFC" />
            <stop offset="70%" stopColor="#E2E8F0" />
            <stop offset="100%" stopColor="#94A3B8" />
          </linearGradient>

          <linearGradient id="glowRadial" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(217, 119, 6, 0.25)" />
            <stop offset="60%" stopColor="rgba(217, 119, 6, 0.08)" />
            <stop offset="100%" stopColor="rgba(217, 119, 6, 0)" />
          </linearGradient>

          <linearGradient id="blueGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(37, 99, 235, 0.18)" />
            <stop offset="100%" stopColor="rgba(37, 99, 235, 0)" />
          </linearGradient>

          <filter id="shadowFilter" x="-10%" y="-10%" width="120%" height="120%">
            <feDropShadow dx="0" dy="4" stdDeviation="6" floodColor="#0F172A" floodOpacity="0.12" />
          </filter>
        </defs>

        {/* Ambient Halo & Rays */}
        <circle cx="260" cy="180" r="160" fill="url(#glowRadial)" className={styles.ambientPulse} />
        <circle cx="260" cy="180" r="130" fill="url(#blueGlow)" />

        {/* Rotating Celestial Judicial Orbit Ring */}
        <g className={styles.rotatingRing}>
          <circle
            cx="260"
            cy="180"
            r="145"
            stroke="#D97706"
            strokeWidth="1.2"
            strokeDasharray="6 8"
            opacity="0.4"
          />
          <circle
            cx="260"
            cy="180"
            r="155"
            stroke="#1D4ED8"
            strokeWidth="1"
            strokeDasharray="2 12"
            opacity="0.3"
          />
          {/* Orbiting star nodes */}
          <circle cx="260" cy="35" r="3.5" fill="#D97706" />
          <circle cx="405" cy="180" r="3.5" fill="#1D4ED8" />
          <circle cx="260" cy="325" r="3.5" fill="#D97706" />
          <circle cx="115" cy="180" r="3.5" fill="#1D4ED8" />
        </g>

        {/* Courtroom Classical Portico Pediment Background */}
        <path
          d="M140 100 L260 40 L380 100 Z"
          fill="none"
          stroke="#CBD5E1"
          strokeWidth="1.5"
          opacity="0.5"
        />
        <line x1="120" y1="102" x2="400" y2="102" stroke="#94A3B8" strokeWidth="2" opacity="0.4" />

        {/* Classical Fluted Pillars in Background */}
        <g opacity="0.25">
          {/* Left Column */}
          <rect x="145" y="105" width="22" height="210" fill="url(#pillarGradient)" rx="2" />
          <line x1="151" y1="110" x2="151" y2="310" stroke="#94A3B8" strokeWidth="1" />
          <line x1="157" y1="110" x2="157" y2="310" stroke="#94A3B8" strokeWidth="1" />
          <rect x="140" y="315" width="32" height="8" fill="#94A3B8" rx="1" />

          {/* Right Column */}
          <rect x="353" y="105" width="22" height="210" fill="url(#pillarGradient)" rx="2" />
          <line x1="359" y1="110" x2="359" y2="310" stroke="#94A3B8" strokeWidth="1" />
          <line x1="365" y1="110" x2="365" y2="310" stroke="#94A3B8" strokeWidth="1" />
          <rect x="348" y="315" width="32" height="8" fill="#94A3B8" rx="1" />
        </g>

        {/* Acoustic Speech Ripples merging into Court Fulcrum */}
        <g className={styles.soundWaves}>
          <path
            d="M 60 180 Q 150 140, 260 180"
            fill="none"
            stroke="url(#brassGradient)"
            strokeWidth="1.8"
            strokeDasharray="4 4"
            className={styles.waveLeft}
          />
          <path
            d="M 460 180 Q 370 140, 260 180"
            fill="none"
            stroke="url(#brassGradient)"
            strokeWidth="1.8"
            strokeDasharray="4 4"
            className={styles.waveRight}
          />
        </g>

        {/* Central Marble & Brass Plinth Base */}
        <g filter="url(#shadowFilter)">
          <rect x="180" y="335" width="160" height="14" rx="4" fill="#1E293B" />
          <rect x="195" y="322" width="130" height="13" rx="3" fill="#334155" />
          <rect x="215" y="312" width="90" height="10" rx="2" fill="url(#goldGradient)" />
        </g>

        {/* Center Vertical Pillar */}
        <g filter="url(#shadowFilter)">
          <rect x="254" y="90" width="12" height="222" fill="url(#brassGradient)" rx="3" />
          {/* Pillar capital rings */}
          <rect x="248" y="120" width="24" height="6" fill="#F59E0B" rx="2" />
          <circle cx="260" cy="180" r="14" fill="url(#goldGradient)" />
          <circle cx="260" cy="180" r="8" fill="#1E293B" />
          <circle cx="260" cy="180" r="4" fill="#F59E0B" />
          {/* Pillar finial top orb */}
          <circle cx="260" cy="85" r="8" fill="url(#brassGradient)" />
          <polygon points="260,65 264,78 256,78" fill="url(#goldGradient)" />
        </g>

        {/* DYNAMIC OSCILLATING SCALES ASSEMBLY */}
        <g className={styles.scalesBeamAssembly}>
          {/* Fulcrum Bearing Pivot */}
          <circle cx="260" cy="98" r="7" fill="url(#goldGradient)" />
          <circle cx="260" cy="98" r="3.5" fill="#1E293B" />

          {/* Balance Beam */}
          <path
            d="M130 102 C190 94, 230 96, 260 98 C290 96, 330 94, 390 102 L388 107 C330 100, 290 102, 260 104 C230 102, 190 100, 132 107 Z"
            fill="url(#goldGradient)"
            filter="url(#shadowFilter)"
          />

          {/* Left Pan Pivot Orb */}
          <circle cx="131" cy="105" r="5" fill="#D97706" />
          {/* Right Pan Pivot Orb */}
          <circle cx="389" cy="105" r="5" fill="#D97706" />

          {/* LEFT SUSPENDED SCALE PAN (Testimony / Local Inference) */}
          <g className={styles.leftPanAssembly}>
            {/* Hanging Chains */}
            <line x1="131" y1="108" x2="95" y2="210" stroke="#B45309" strokeWidth="1.5" strokeDasharray="3 3" />
            <line x1="131" y1="108" x2="131" y2="210" stroke="#B45309" strokeWidth="1.5" strokeDasharray="3 3" />
            <line x1="131" y1="108" x2="167" y2="210" stroke="#B45309" strokeWidth="1.5" strokeDasharray="3 3" />

            {/* Left Brass Pan */}
            <path
              d="M90 210 Q131 245, 172 210 Z"
              fill="url(#goldGradient)"
              filter="url(#shadowFilter)"
            />
            {/* Pan Rim */}
            <ellipse cx="131" cy="210" rx="41" ry="5" fill="#FDE68A" opacity="0.8" />

            {/* Glowing Testimony Wave inside left pan */}
            <g className={styles.speechOrb}>
              <circle cx="131" cy="195" r="14" fill="rgba(37, 99, 235, 0.15)" />
              <circle cx="131" cy="195" r="8" fill="#2563EB" opacity="0.85" />
              <text x="131" y="198" textAnchor="middle" fontSize="9" fill="#FFFFFF" fontWeight="700">🎤</text>
            </g>
          </g>

          {/* RIGHT SUSPENDED SCALE PAN (Privacy Barrier / L2 DP Bound) */}
          <g className={styles.rightPanAssembly}>
            {/* Hanging Chains */}
            <line x1="389" y1="108" x2="353" y2="210" stroke="#B45309" strokeWidth="1.5" strokeDasharray="3 3" />
            <line x1="389" y1="108" x2="389" y2="210" stroke="#B45309" strokeWidth="1.5" strokeDasharray="3 3" />
            <line x1="389" y1="108" x2="425" y2="210" stroke="#B45309" strokeWidth="1.5" strokeDasharray="3 3" />

            {/* Right Brass Pan */}
            <path
              d="M348 210 Q389 245, 430 210 Z"
              fill="url(#goldGradient)"
              filter="url(#shadowFilter)"
            />
            {/* Pan Rim */}
            <ellipse cx="389" cy="210" rx="41" ry="5" fill="#FDE68A" opacity="0.8" />

            {/* Glowing Privacy Shield inside right pan */}
            <g className={styles.privacyOrb}>
              <circle cx="389" cy="195" r="14" fill="rgba(16, 185, 129, 0.15)" />
              <circle cx="389" cy="195" r="8" fill="#10B981" opacity="0.85" />
              <text x="389" y="198" textAnchor="middle" fontSize="9" fill="#FFFFFF" fontWeight="700">🛡️</text>
            </g>
          </g>
        </g>

        {/* Tamil Judicial Inscription Ribbon Base */}
        <g filter="url(#shadowFilter)">
          <path
            d="M 170 355 L 350 355 L 340 372 L 180 372 Z"
            fill="#FEF3C7"
            stroke="#D97706"
            strokeWidth="1.2"
          />
          <text
            x="260"
            y="368"
            textAnchor="middle"
            fill="#92400E"
            fontSize="10"
            fontWeight="700"
            letterSpacing="0.08em"
            fontFamily="serif"
          >
            வாய்மையே வெல்லும் • TRUTH ALONE TRIUMPHS
          </text>
        </g>
      </svg>
      <div className={styles.captionBadge}>
        <span className={styles.badgeDot} />
        <span>Judicial Equivalence: <strong>Local Speech Inference</strong> balanced with <strong>Differential Privacy</strong></span>
      </div>
    </div>
  );
}
