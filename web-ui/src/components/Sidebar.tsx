import { NavLink, useLocation } from 'react-router-dom'
import './Sidebar.css'

interface NavItem {
  to: string
  label: string
  icon: string
}

interface NavGroup {
  group: string
  icon: string
  items: NavItem[]
}

const NAV_ITEMS: NavGroup[] = [
  {
    group: 'Use Case 1',
    icon: '⚡',
    items: [
      { to: '/uc1/intro', label: 'Giới thiệu', icon: '📋' },
      { to: '/uc1/demo', label: 'Real-time Demo', icon: '📡' },
    ],
  },
  {
    group: 'Use Case 2',
    icon: '💪',
    items: [
      { to: '/uc2/intro', label: 'Giới thiệu', icon: '📋' },
      { to: '/uc2/demo', label: 'Phục hồi cơ', icon: '📊' },
    ],
  },
]

export default function Sidebar() {
  const location = useLocation()
  const isIntro = location.pathname.includes('/intro')

  return (
    <aside className={`sidebar ${isIntro ? 'sidebar--hidden' : ''}`}>
      <div className="sidebar__logo">
        <div className="sidebar__logo-icon">
          <span className="sidebar__logo-dot" />
        </div>
        <div>
          <div className="sidebar__logo-title">sEMG Platform</div>
          <div className="sidebar__logo-sub">Fatigue Analysis</div>
        </div>
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map(group => (
          <div key={group.group} className="sidebar__group">
            <div className="sidebar__group-label">
              <span>{group.icon}</span> {group.group}
            </div>
            {group.items.map(item => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
                }
              >
                <span className="sidebar__link-icon">{item.icon}</span>
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar__footer">
        <div className="sidebar__footer-text">Dữ liệu demo — mô phỏng</div>
        <div className="sidebar__footer-text">Không phải bệnh nhân thật</div>
      </div>
    </aside>
  )
}
