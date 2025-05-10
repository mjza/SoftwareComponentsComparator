package ca.devsearch.BatchedIssueTransfer;

import java.sql.*;
import java.util.Properties;
import io.github.cdimascio.dotenv.Dotenv;

public class App {

	public static void main(String[] args) throws Exception {

		Dotenv dotenv = Dotenv.load();

		Properties props = new Properties();
		props.setProperty("user", dotenv.get("DB_USER"));
		props.setProperty("password", dotenv.get("DB_PASSWORD"));

		String url = String.format("jdbc:postgresql://%s:%s/%s", dotenv.get("DB_HOST"), dotenv.get("DB_PORT"),
				dotenv.get("DB_NAME"));

		Connection conn = DriverManager.getConnection(url, props);

		int batchSize = 1000;
		long lastMaxId = 0;

		try (Statement stmt = conn.createStatement();
				ResultSet rs = stmt.executeQuery("SELECT COALESCE(MAX(issue_id), 0) FROM combined_issues")) {
			if (rs.next()) {
				lastMaxId = rs.getLong(1);
				System.out.println("▶️ Starting from issue_id > " + lastMaxId);
			}
		}
		
		long totalSourceRows = 0;
		String countSourceSql = """
		    SELECT COUNT(DISTINCT i.issue_id)
		    FROM issues i
		    WHERE i.body_text IS NOT NULL
		      AND char_length(i.body_text) > 0
		      AND i.owner_type = 'User'
		""";

		try (Statement stmt = conn.createStatement();
		     ResultSet rs = stmt.executeQuery(countSourceSql)) {
		    if (rs.next()) {
		        totalSourceRows = rs.getLong(1);
		        System.out.println("📦 Total source rows: " + totalSourceRows);
		    }
		}
		
		long existingRows = 0;
		try (Statement stmt = conn.createStatement();
		     ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM combined_issues")) {
		    if (rs.next()) {
		        existingRows = rs.getLong(1);
		        System.out.println("✅ Already transferred: " + existingRows);
		    }
		}
		
		System.out.printf("➡️ Remaining to transfer: %d%n", totalSourceRows - existingRows);


		while (true) {
			String sql = "WITH filtered AS ( "
					+ "SELECT i.issue_id, i.project_id, i.number, i.title, i.state, i.comments, i.body_text, "
					+ "c.id AS comment_id, c.body_text AS comment_body_text, i.created_at " + "FROM issues i "
					+ "LEFT JOIN comments c ON c.issue_id = i.issue_id "
					+ "WHERE i.body_text IS NOT NULL AND char_length(i.body_text) > 0 "
					+ "AND i.owner_type = 'User' AND i.issue_id > ? ) "
					+ "SELECT issue_id, project_id, number, state, comments, "
					+ "title || E'\\n\\n' || body_text || COALESCE(E'\\n\\n' || STRING_AGG(comment_body_text, E'\\n\\n' ORDER BY comment_id), '') AS issue_text, created_at "
					+ "FROM filtered "
					+ "GROUP BY issue_id, project_id, number, title, state, comments, body_text, created_at "
					+ "ORDER BY issue_id LIMIT ?";

			try (PreparedStatement ps = conn.prepareStatement(sql)) {
				ps.setLong(1, lastMaxId);
				ps.setInt(2, batchSize);
				ResultSet rs = ps.executeQuery();

				int count = 0;

				// Prepare insert statement once (outside loop for efficiency)
				String insertSql = "INSERT INTO public.combined_issues "
						+ "(issue_id, project_id, number, state, comments, issue_text, created_at) "
						+ "VALUES (?, ?, ?, ?, ?, ?, ?)";
				try (PreparedStatement insertStmt = conn.prepareStatement(insertSql)) {
					while (rs.next()) {
				        long issueId = rs.getLong("issue_id");
				        String issueText = rs.getString("issue_text");

				        insertStmt.setLong(1, issueId);
				        insertStmt.setLong(2, rs.getLong("project_id"));
				        insertStmt.setInt(3, rs.getInt("number"));
				        insertStmt.setString(4, rs.getString("state"));
				        insertStmt.setInt(5, rs.getInt("comments"));
				        insertStmt.setString(6, issueText);
				        insertStmt.setTimestamp(7, rs.getTimestamp("created_at"));

				        insertStmt.addBatch(); // enqueue

				        lastMaxId = issueId;
				        count++;
				    }
				    insertStmt.executeBatch(); // execute all at once
				    existingRows += count;
				    System.out.printf("➡️ Remaining to transfer: %d%n", totalSourceRows - existingRows);

				}


				if (count < batchSize) {
					System.out.println("✅ Done: no more data.");
					break;
				}
			}

		}

		conn.close();
	}
}
