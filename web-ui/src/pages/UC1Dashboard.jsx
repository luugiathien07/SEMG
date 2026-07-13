import { motion } from 'framer-motion'

export default function UC1Dashboard() {
  return (
    <div style={{ padding: '48px', maxWidth: 1100, margin: '0 auto' }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 style={{ fontSize: 'var(--font-2xl)', fontWeight: 800, marginBottom: 16 }}>
          ⚡ Use Case 1: Giám sát Mỏi cơ Real-time
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-md)', marginBottom: 32 }}>
          Đang phát triển — chức năng Real-time EMG sẽ được tích hợp tại đây.
        </p>

        <div className="glass-card" style={{ padding: 48, textAlign: 'center' }}>
          <div style={{ fontSize: '4rem', marginBottom: 16 }}>🚧</div>
          <h3 style={{ fontSize: 'var(--font-xl)', marginBottom: 12 }}>Coming Soon</h3>
          <p style={{ color: 'var(--text-secondary)', maxWidth: 500, margin: '0 auto', lineHeight: 1.6 }}>
            Real-time EMG waveform animation, phân loại mỏi từng đoạn,
            biểu đồ RMS & MDF theo thời gian, và grid 64 kênh sẽ được
            tích hợp trong phiên bản tiếp theo.
          </p>
        </div>
      </motion.div>
    </div>
  )
}
