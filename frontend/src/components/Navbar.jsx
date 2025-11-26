import { NavLink } from 'react-router-dom'

export default function Navbar() {
  return (
    <div className="navbar">
      <div className="wrap">
        <div className="brand">Audio RAG Chatbot</div>
        <div className="navlinks">
          <NavLink to="/upload">Upload</NavLink>
          <NavLink to="/chat">Chat</NavLink>
        </div>
      </div>
    </div>
  )
}
