import duckdb

def main():
    con = duckdb.connect("./annotations_per_tile.duckdb")

    # for table in ["column_dictionary", "export_metadata", "tiles"]:
    #     print(f"--- Showing Table {table} --- ")
    #     con.sql(f"""
    #         DESCRIBE {table}
    #     """).show()
    #     print("\n")

    # --- export_metadata ---
    # con.sql("""
    #     SELECT key, value
    #     FROM export_metadata
    # """).show()

    # --- column_dictionary ---
    # con.sql("""
    #     SELECT column_name, data_type, source_field, explanation
    #     FROM column_dictionary
    #     LIMIT 10
    # """).show()

    # --- tiles ---
    # print(con.table("tiles").columns)
    con.sql("""
        DESCRIBE tiles
    """).show()
    
    # result = con.execute("DESCRIBE  tiles").fetchall()
    # print(result)

if __name__ == "__main__":
    main()
