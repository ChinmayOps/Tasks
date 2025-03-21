#AUTOMATED REPORT GENERATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

class SimpleReportGenerator:
    def __init__(self):
        """Initialize the simplified report generator."""
        self.output_dir = "reports"
        self.charts_dir = os.path.join(self.output_dir, "charts")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self.title_style = self.styles['Heading1']
        self.subtitle_style = self.styles['Heading2']
        self.normal_style = self.styles['Normal']
        self.caption_style = ParagraphStyle(
            'Caption', parent=self.styles['Italic'],
            fontSize=9, textColor=colors.darkgrey, alignment=1
        )
        
    def generate_sample_data(self):
        """Generate sample sales data."""
        np.random.seed(42)
        
        # Define parameters
        products = ['Product A', 'Product B', 'Product C', 'Product D']
        regions = ['North', 'South', 'East', 'West']
        
        # Generate date range for the past 6 months
        dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
        
        # Create sample data
        num_records = 500
        
        data = {
            'Date': np.random.choice(dates, num_records),
            'Product': np.random.choice(products, num_records),
            'Region': np.random.choice(regions, num_records),
            'Units': np.random.randint(1, 50, num_records),
            'Price': np.random.uniform(10, 500, num_records).round(2)
        }
        
        # Calculate revenue
        data['Revenue'] = data['Units'] * data['Price']
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values('Date')
        
        # Save to CSV
        file_path = os.path.join(self.output_dir, "sample_sales_data.csv")
        df.to_csv(file_path, index=False)
        
        print(f"Sample data generated and saved to {file_path}")
        return df
    
    def analyze_data(self, data):
        """Perform basic analysis on the data."""
        # Summary statistics
        analysis = {
            'total_revenue': data['Revenue'].sum(),
            'total_units': data['Units'].sum(),
            'avg_price': data['Price'].mean()
        }
        
        # Monthly trends
        data['Month'] = data['Date'].dt.to_period('M')
        monthly_data = data.groupby('Month').agg({
            'Revenue': 'sum',
            'Units': 'sum'
        }).reset_index()
        monthly_data['Month'] = monthly_data['Month'].astype(str)
        analysis['monthly_data'] = monthly_data
        
        # Product performance
        product_data = data.groupby('Product').agg({
            'Revenue': 'sum',
            'Units': 'sum'
        }).sort_values('Revenue', ascending=False).reset_index()
        analysis['product_data'] = product_data
        
        # Regional performance
        region_data = data.groupby('Region').agg({
            'Revenue': 'sum'
        }).sort_values('Revenue', ascending=False).reset_index()
        analysis['region_data'] = region_data
        
        return analysis
    
    def create_charts(self, analysis):
        """Create basic charts for the report."""
        chart_files = {}
        
        # Monthly revenue trend
        plt.figure(figsize=(8, 4))
        plt.plot(analysis['monthly_data']['Month'], analysis['monthly_data']['Revenue'], marker='o')
        plt.title('Monthly Revenue Trend')
        plt.xlabel('Month')
        plt.ylabel('Revenue ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        monthly_chart_path = os.path.join(self.charts_dir, 'monthly_revenue.png')
        plt.savefig(monthly_chart_path, dpi=200)
        plt.close()
        chart_files['monthly_revenue'] = monthly_chart_path
        
        # Product revenue
        plt.figure(figsize=(8, 4))
        plt.bar(analysis['product_data']['Product'], analysis['product_data']['Revenue'])
        plt.title('Revenue by Product')
        plt.xlabel('Product')
        plt.ylabel('Revenue ($)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        product_chart_path = os.path.join(self.charts_dir, 'product_revenue.png')
        plt.savefig(product_chart_path, dpi=200)
        plt.close()
        chart_files['product_revenue'] = product_chart_path
        
        # Regional performance
        plt.figure(figsize=(8, 4))
        plt.pie(analysis['region_data']['Revenue'], labels=analysis['region_data']['Region'], 
                autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Revenue Distribution by Region')
        plt.tight_layout()
        
        region_chart_path = os.path.join(self.charts_dir, 'region_revenue.png')
        plt.savefig(region_chart_path, dpi=200)
        plt.close()
        chart_files['region_revenue'] = region_chart_path
        
        return chart_files
    
    def create_report(self, analysis, chart_files, output_file="sales_report.pdf"):
        """Create a simplified PDF report."""
        doc_path = os.path.join(self.output_dir, output_file)
        doc = SimpleDocTemplate(doc_path, pagesize=letter, 
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        elements = []
        
        # Title and Date
        elements.append(Paragraph("Sales Performance Report", self.title_style))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", self.subtitle_style))
        
        summary_text = f"""
        <b>Total Revenue:</b> ${analysis['total_revenue']:,.2f}
        <br/>
        <b>Total Units Sold:</b> {analysis['total_units']:,}
        <br/>
        <b>Average Price:</b> ${analysis['avg_price']:.2f}
        """
        
        elements.append(Paragraph(summary_text, self.normal_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Monthly Trend Analysis
        elements.append(Paragraph("Monthly Revenue Trend", self.subtitle_style))
        
        if 'monthly_revenue' in chart_files:
            img = Image(chart_files['monthly_revenue'], width=6*inch, height=3*inch)
            elements.append(img)
            elements.append(Paragraph("Figure 1: Monthly Revenue Trend", self.caption_style))
        
        elements.append(Spacer(1, 0.25*inch))
        
        # Product Performance
        elements.append(Paragraph("Product Performance", self.subtitle_style))
        
        if 'product_revenue' in chart_files:
            img = Image(chart_files['product_revenue'], width=6*inch, height=3*inch)
            elements.append(img)
            elements.append(Paragraph("Figure 2: Revenue by Product", self.caption_style))
        
        elements.append(Spacer(1, 0.25*inch))
        
        # Top products table
        table_data = [['Product', 'Revenue', 'Units Sold']]
        
        for _, row in analysis['product_data'].iterrows():
            table_data.append([
                row['Product'],
                f"${row['Revenue']:,.2f}",
                f"{row['Units']:,}"
            ])
        
        product_table = Table(table_data, colWidths=[2*inch, 2*inch, 2*inch])
        product_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ]))
        
        elements.append(product_table)
        elements.append(Paragraph("Table 1: Product Performance", self.caption_style))
        
        elements.append(PageBreak())
        
        # Regional Analysis
        elements.append(Paragraph("Regional Analysis", self.subtitle_style))
        
        if 'region_revenue' in chart_files:
            img = Image(chart_files['region_revenue'], width=6*inch, height=3*inch)
            elements.append(img)
            elements.append(Paragraph("Figure 3: Revenue by Region", self.caption_style))
        
        elements.append(Spacer(1, 0.25*inch))
        
        # Conclusion
        elements.append(Paragraph("Conclusion", self.subtitle_style))
        
        conclusion_text = f"""
        This analysis shows a total revenue of ${analysis['total_revenue']:,.2f} across all products and regions.
        The top-performing product is {analysis['product_data'].iloc[0]['Product']} with 
        ${analysis['product_data'].iloc[0]['Revenue']:,.2f} in revenue.
        The highest revenue region is {analysis['region_data'].iloc[0]['Region']}.
        """
        
        elements.append(Paragraph(conclusion_text, self.normal_style))
        
        # Build the PDF
        doc.build(elements)
        print(f"Report generated successfully: {doc_path}")
        
        return doc_path
    
    def generate_report(self, output_file="sales_report.pdf"):
        """Main function to generate the complete report."""
        # Generate sample data
        data = self.generate_sample_data()
        
        # Analyze data
        analysis = self.analyze_data(data)
        
        # Create charts
        chart_files = self.create_charts(analysis)
        
        # Generate report
        report_path = self.create_report(analysis, chart_files, output_file)
        
        return report_path

def main():
    try:
        # Create and run the report generator
        report_gen = SimpleReportGenerator()
        report_path = report_gen.generate_report()
        
        print(f"Report successfully generated at: {report_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()