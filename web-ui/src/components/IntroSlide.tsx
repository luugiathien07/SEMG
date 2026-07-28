import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import type { IntroContent } from '../types'
import './IntroSlide.css'

function parseBold(text: string): ReactNode[] {
  const parts = text.split(/\*\*(.*?)\*\*/g)
  return parts.map((part, i) =>
    i % 2 === 1 ? <strong key={i}>{part}</strong> : part
  )
}

interface IntroSlideProps {
  content: IntroContent
}

export default function IntroSlide({ content }: IntroSlideProps) {
  const navigate = useNavigate()

  return (
    <div className="intro-slide">
      <div className="intro-slide__bg" />
      <motion.div
        className="intro-slide__content"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
      >
        {/* Badge */}
        <div className="intro-slide__badge">
          <span className="intro-slide__badge-dot" />
          {content.badge}
        </div>

        {/* Title */}
        <h1 className="intro-slide__title text-gradient">{content.title}</h1>
        <p className="intro-slide__subtitle">{content.subtitle}</p>

        {/* 4-column grid */}
        <div className="intro-slide__grid">
          {content.cards.map((card, i) => (
            <motion.div
              key={i}
              className="intro-slide__card glass-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + i * 0.1, duration: 0.5 }}
            >
              <div className={`intro-slide__card-icon intro-slide__card-icon--${i}`}>
                {card.icon}
              </div>
              <div className="intro-slide__card-label">{card.label}</div>
              <p className="intro-slide__card-text">{parseBold(card.text)}</p>
            </motion.div>
          ))}
        </div>

        {/* Output tags */}
        <div className="intro-slide__outputs">
          {content.outputs.map((out, i) => (
            <motion.div
              key={i}
              className="intro-slide__output-tag"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 + i * 0.08, duration: 0.3 }}
            >
              <span className="intro-slide__check">✓</span> {out}
            </motion.div>
          ))}
        </div>

        {/* CTA Button */}
        <motion.button
          className="intro-slide__cta"
          onClick={() => navigate(content.demoPath)}
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.97 }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1, duration: 0.4 }}
        >
          Bắt đầu Demo →
        </motion.button>

        {/* Footer */}
        <p className="intro-slide__footer">{content.footer}</p>
      </motion.div>
    </div>
  )
}
