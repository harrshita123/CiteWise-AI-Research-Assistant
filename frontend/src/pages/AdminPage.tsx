import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getAdminUsers, getCurrentUser, SignedInUser } from "../services/authApi";

export default function AdminPage() {
  const [admin, setAdmin] = useState<SignedInUser | null>(null);
  const [users, setUsers] = useState<SignedInUser[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let isActive = true;

    async function loadDashboard() {
      try {
        const currentUser = await getCurrentUser();

        if (currentUser.role !== "admin") {
          throw new Error("Administrator access is required.");
        }

        const userList = await getAdminUsers();

        if (isActive) {
          setAdmin(currentUser);
          setUsers(userList);
        }
      } catch (error) {
        if (isActive) {
          setError(error instanceof Error ? error.message : "The admin dashboard could not be loaded.");
        }
      }
    }

    void loadDashboard();

    return () => {
      isActive = false;
    };
  }, []);

  return (
    <div className="admin-page">
      <main className="admin-main">
        <p className="label">Administration</p>
        <h1>User access</h1>

        {error ? (
          <div className="admin-error">
            <p>{error}</p>
            <Link className="primary-button small" to="/login">Log in as admin</Link>
          </div>
        ) : !admin ? (
          <p className="admin-loading">Checking administrator access...</p>
        ) : (
          <>
            <div className="admin-summary">
              <p>Signed in as <strong>{admin.email}</strong></p>
              <span>{users.length} {users.length === 1 ? "account" : "accounts"}</span>
            </div>

            <div className="admin-table-wrap">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Role</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => (
                    <tr key={user.id}>
                      <td>{user.name || "Not provided"}</td>
                      <td>{user.email}</td>
                      <td><span className={`role-badge ${user.role}`}>{user.role}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
