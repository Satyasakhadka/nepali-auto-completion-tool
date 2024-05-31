import React from "react";
import { Link } from "react-router-dom";
import "./navbar.css";

const Navbar = () => {
  return (
    <div className="nav">
      <div className="nav-logo">लिपि <span className="fancy-text">सहायक</span></div>
      <ul className="nav-menu">
        <Link to="/" className="nav-link">Home</Link>
        <Link to="/explore" className="nav-link">TextEditor</Link>
      </ul>
    </div>
  );
};

export default Navbar;
