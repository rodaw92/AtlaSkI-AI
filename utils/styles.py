def local_css():
    return """
    <style>
        /* Main page styling */
        .main {
            background-color: #f9fafb;
        }
        
        /* Header styling */
        .header-container {
            background-color: #1e3a8a;
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-image: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .header-subtitle {
            font-size: 1.2rem;
            opacity: 0.8;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1e3a8a;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 0.5rem;
        }
        
        /* Status indicators */
        .status-verified {
            display: inline-block;
            background-color: #10b981;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-rejected {
            display: inline-block;
            background-color: #ef4444;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .status-waiting {
            display: inline-block;
            background-color: #f59e0b;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Quality indicator colors */
        .quality-high {
            color: #10b981;
            font-weight: 600;
        }
        
        .quality-medium {
            color: #3b82f6;
            font-weight: 600;
        }
        
        .quality-spatial {
            color: #f59e0b;
            font-weight: 600;
        }
        
        .quality-external {
            color: #8b5cf6;
            font-weight: 600;
        }
        
        .quality-semantic {
            color: #06b6d4;
            font-weight: 600;
        }
        
        .quality-low {
            color: #ef4444;
            font-weight: 600;
        }
        
        /* Alert boxes */
        .alert-info {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .alert-warning {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .alert-success {
            background-color: #ecfdf5;
            border-left: 4px solid #10b981;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        .alert-error {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 4px;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #1e293b;
            color: white;
        }
        
        /* Improve button styling */
        .stButton>button {
            width: 100%;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        /* Data tables */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .dataframe th {
            background-color: #f3f4f6;
            padding: 0.75rem;
            text-align: left;
            font-weight: 600;
            color: #374151;
        }
        
        .dataframe td {
            padding: 0.75rem;
            border-top: 1px solid #e5e7eb;
        }
        
        .dataframe tr:hover {
            background-color: #f9fafb;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            font-size: 14px;
            font-weight: 500;
            background-color: #f3f4f6;
            border-radius: 6px 6px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white;
            border-top: 3px solid #1e3a8a;
        }
    </style>
    """ 