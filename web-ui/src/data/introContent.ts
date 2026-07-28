import type { IntroContent } from '../types'

export const UC1_INTRO: IntroContent = {
  badge: 'sEMG Fatigue Analysis Platform',
  title: 'Use Case 1: Giám sát Mỏi cơ Real-time',
  subtitle: 'Real-time Fatigue Monitoring — Phát hiện mỏi cơ trong buổi tập bằng tín hiệu sEMG',
  cards: [
    {
      icon: '🎯',
      label: 'Mục tiêu',
      text: 'Phát hiện **mỏi cơ theo thời gian thực** trong buổi tập, cảnh báo sớm cho KTV/bác sĩ trước khi bệnh nhân kiệt sức.',
    },
    {
      icon: '⚙️',
      label: 'Chức năng',
      text: 'Trích xuất 14 đặc trưng từ HD-sEMG → so sánh 5 mô hình ML → **phân loại mỏi/chưa mỏi** từng đoạn tín hiệu.',
    },
    {
      icon: '📊',
      label: 'Dữ liệu',
      text: 'HD-sEMG **64 kênh**, 2048 Hz. 4 đối tượng, 6 mức %MVC. Dữ liệu thu tại phòng thí nghiệm.',
    },
    {
      icon: '📋',
      label: 'Ý nghĩa lâm sàng',
      text: 'Hỗ trợ KTV **ra quyết định dừng tập** hoặc giảm tải kịp thời dựa trên bằng chứng sinh lý khách quan.',
    },
  ],
  outputs: [
    'Sóng EMG real-time',
    'Phân loại mỏi từng đoạn',
    'RMS & MDF theo thời gian',
    'Grid 64 kênh',
  ],
  footer: 'Dữ liệu demo: thu thực tế · Mô hình: LogisticRegression (F1 = 0.830) · Kết quả hỗ trợ đánh giá, không thay thế chẩn đoán',
  demoPath: '/uc1/demo',
  accentColor: 'coral',
}

export const UC2_INTRO: IntroContent = {
  badge: 'sEMG Fatigue Analysis Platform',
  title: 'Use Case 2: Giám sát Phục hồi Cơ',
  subtitle: 'Muscle Recovery Monitoring — Theo dõi sức bền và phục hồi qua nhiều buổi trị liệu',
  cards: [
    {
      icon: '🎯',
      label: 'Mục tiêu',
      text: 'Đánh giá **cơ có thực sự khoẻ hơn** qua từng buổi trị liệu — đo bằng chỉ số khách quan từ tín hiệu sEMG.',
    },
    {
      icon: '⚙️',
      label: 'Chức năng',
      text: 'Tính **P(mỏi)** bằng KNN 7 đặc trưng. So sánh **sức bền** và **chỉ số đối xứng lành–liệt** qua các buổi.',
    },
    {
      icon: '📊',
      label: 'Dữ liệu',
      text: '**HD-sEMG 64 kênh**, 2000 Hz. Bộ dữ liệu PhysioMio bệnh nhân đột quỵ + mô phỏng co cơ duy trì.',
    },
    {
      icon: '📋',
      label: 'Ý nghĩa lâm sàng',
      text: 'Hỗ trợ KTV phục hồi chức năng **theo dõi tiến triển** và điều chỉnh phác đồ trị liệu theo bằng chứng.',
    },
  ],
  outputs: [
    'Biểu đồ P(mỏi) so sánh 2 buổi',
    'Điểm sức bền qua 8 buổi',
    'Symmetry Index lành–liệt',
    'Khám phá tín hiệu gốc',
  ],
  footer: 'Dữ liệu demo: mô phỏng + PhysioMio · Mô hình: KNN (k=7, F1 = 0.928) · Kết quả hỗ trợ đánh giá, không thay thế chẩn đoán',
  demoPath: '/uc2/demo',
  accentColor: 'teal',
}
