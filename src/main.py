from visualization import create_app

app = create_app()

if __name__ == '__main__':
    # debug=true (after a change re-runs automatically)
    app.run(debug=True)
